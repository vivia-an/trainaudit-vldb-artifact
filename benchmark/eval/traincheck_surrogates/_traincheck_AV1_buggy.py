from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_AV1/trace_buggy"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_AV1/trace_buggy/proxy_log.json'})

from traincheck.proxy_wrapper.proxy import Proxy

import glob
import importlib
from traincheck.proxy_wrapper.proxy_config import auto_observer_config
spec = importlib.util.find_spec('traincheck')
if spec and spec.origin:
    traincheck_folder = os.path.dirname(spec.origin)
    print("traincheck folder: ", traincheck_folder)
else:
    raise Exception("traincheck is not installed properly")
print("auto observer enabled with observing depth: ", auto_observer_config["enable_auto_observer_depth"])
enable_auto_observer_depth = auto_observer_config["enable_auto_observer_depth"]
neglect_hidden_func = auto_observer_config["neglect_hidden_func"]
neglect_hidden_module = auto_observer_config["neglect_hidden_module"]
observe_then_unproxy = auto_observer_config["observe_then_unproxy"]
observe_up_to_depth = auto_observer_config["observe_up_to_depth"]
if observe_up_to_depth:
    print("observe up to the depth of the function call")
else:
    print("observe only the function call at the depth")
from traincheck.static_analyzer.graph_generator.call_graph_parser import add_observer_given_call_graph

log_files = glob.glob(
    os.path.join(traincheck_folder, "static_analyzer", "func_level", "*.log")
)
print("log_files: ", log_files)
for log_file in log_files:
    add_observer_given_call_graph(
        log_file,
        depth=enable_auto_observer_depth,
        observe_up_to_depth=observe_up_to_depth,
        neglect_hidden_func=neglect_hidden_func,
        neglect_hidden_module=neglect_hidden_module,
        observe_then_unproxy=observe_then_unproxy,
    )
'AV1 surrogate (buggy): fused layernorm impl differs from unfused by ~1e-3.\n\nTriggers P12 Algorithm Variant / Formula Equivalence: fused and unfused\nimplementations of the same operator should produce numerically equivalent\noutput (within fp32 tolerance ~1e-6). Buggy fused path drops the eps term.\n'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

class BuggyFusedLayerNorm(nn.Module):
    """Buggy fused impl: drops eps, uses 1/std instead of 1/sqrt(var+eps)."""

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / std + self.bias

class ReferenceLayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

def main():
    annotate_stage('init')
    import torch.nn as _nn
    from traincheck.instrumentor.tracer import Instrumentor
    Instrumentor(_nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
    model = _nn.Linear(1, 1)
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    torch.manual_seed(0)
    fused = BuggyFusedLayerNorm(64)
    ref = ReferenceLayerNorm(64)
    annotate_stage('testing')
    with torch.no_grad():
        ref.weight.copy_(fused.weight)
        ref.bias.copy_(fused.bias)
    diffs = []
    for step in range(20):
        x = torch.randn(4, 64)
        out_fused = fused(x)
        out_ref = ref(x)
        rel_diff = (out_fused - out_ref).abs().max() / out_ref.abs().max()
        diffs.append(rel_diff.item())
    print(f'[AV1_buggy] avg fused-vs-ref rel_diff = {sum(diffs) / len(diffs):.6e}')
    print(f'[AV1_buggy] max rel_diff = {max(diffs):.6e}  (P12 threshold: 1e-5)')
    return diffs
if __name__ == '__main__':
    main()