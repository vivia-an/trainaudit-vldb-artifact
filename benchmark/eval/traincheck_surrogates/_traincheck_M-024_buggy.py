from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_M-024/trace_buggy"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_M-024/trace_buggy/proxy_log.json'})

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
'M-024 surrogate (buggy): apply_input_jitter uses torch.distributions.Uniform\nwhich silently promotes bf16 → fp32 → router input dtype mismatch.\nMirrors Megatron `20b395424d` regression.\n'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

class JitterRouter(nn.Module):

    def __init__(self, hidden=8, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts).to(torch.bfloat16)

    def forward(self, x):
        x_bf = x.to(torch.bfloat16)
        dist = torch.distributions.Uniform(torch.tensor(-0.01), torch.tensor(0.01))
        jitter = dist.sample(x_bf.shape)
        promoted = x_bf + jitter
        return self.linear(promoted.to(torch.bfloat16))

def main():
    annotate_stage('init')
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), JitterRouter(8, 4))
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    opt = optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        annotate_stage('training')
        opt.step()
if __name__ == '__main__':
    main()