from traincheck import annotate_stage
import os
os.environ.setdefault('ML_DAIKON_OUTPUT_DIR', "/tmp/tc_OF1/trace_buggy")
os.makedirs(os.environ['ML_DAIKON_OUTPUT_DIR'], exist_ok=True)

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': os.path.join(os.environ['ML_DAIKON_OUTPUT_DIR'], 'proxy_log.json')})

from traincheck.proxy_wrapper.proxy import Proxy

import glob
import importlib
from traincheck.proxy_wrapper.proxy_config import auto_observer_config
spec = importlib.util.find_spec('traincheck')
if spec and spec.origin:
    traincheck_folder = os.path.dirname(spec.origin)
else:
    raise Exception("traincheck is not installed properly")
enable_auto_observer_depth = auto_observer_config["enable_auto_observer_depth"]
neglect_hidden_func = auto_observer_config["neglect_hidden_func"]
neglect_hidden_module = auto_observer_config["neglect_hidden_module"]
observe_then_unproxy = auto_observer_config["observe_then_unproxy"]
observe_up_to_depth = auto_observer_config["observe_up_to_depth"]
from traincheck.static_analyzer.graph_generator.call_graph_parser import add_observer_given_call_graph

log_files = glob.glob(
    os.path.join(traincheck_folder, "static_analyzer", "func_level", "*.log")
)
for log_file in log_files:
    add_observer_given_call_graph(
        log_file,
        depth=enable_auto_observer_depth,
        observe_up_to_depth=observe_up_to_depth,
        neglect_hidden_func=neglect_hidden_func,
        neglect_hidden_module=neglect_hidden_module,
        observe_then_unproxy=observe_then_unproxy,
    )
'OF1 surrogate (buggy): CPU-offload optimizer state restored at fp16.'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()


def fake_offload_then_restore(t, restore_dtype):
    cpu_copy = t.detach().clone()
    return cpu_copy.to(restore_dtype).to(t.dtype)


def main():
    annotate_stage('init')
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
    opt = optim.AdamW(model.parameters(), lr=0.001)
    annotate_stage('testing')
    for step in range(20):
        opt.zero_grad()
        x = torch.randn(4, 16)
        y = model(x).pow(2).sum()
        y.backward()
        gnorm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
        # buggy: fp16 round-trip during offload restore
        gnorm_after_offload = fake_offload_then_restore(gnorm, restore_dtype=torch.float16)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gnorm_after_offload.item() * 0.99)
        annotate_stage('training')
        opt.step()


if __name__ == '__main__':
    main()
