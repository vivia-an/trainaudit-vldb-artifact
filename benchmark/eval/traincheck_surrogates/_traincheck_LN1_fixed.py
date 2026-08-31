from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_LN1/trace_fixed"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_LN1/trace_fixed/proxy_log.json'})

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
'LN1 surrogate (fixed): aux_loss correctly divided by num_valid_tokens.'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

def compute_aux_loss(logits, mask, mode, micro_batch):
    raw = (logits * logits).sum()
    if mode == 'buggy':
        return raw / micro_batch
    else:
        return raw / mask.sum().clamp(min=1)

def main():
    annotate_stage('init')
    import torch.nn as _nn
    from traincheck.instrumentor.tracer import Instrumentor
    Instrumentor(_nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
    model = _nn.Linear(1, 1)
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    torch.manual_seed(0)
    micro_batch, seq_len, vocab = (4, 16, 32)
    head = nn.Linear(8, vocab)
    losses = []
    for step in range(20):
        x = torch.randn(micro_batch, seq_len, 8)
        logits = head(x)
        mask = torch.ones(micro_batch, seq_len, dtype=torch.bool)
        aux = compute_aux_loss(logits, mask, 'fixed', micro_batch)
        losses.append(aux.item())
    print(f'[LN1_fixed] micro_batch={micro_batch}  seq_len={seq_len}  → divisor used: {micro_batch * seq_len}')
    print(f'[LN1_fixed] avg aux_loss = {sum(losses) / len(losses):.4f}')
if __name__ == '__main__':
    main()