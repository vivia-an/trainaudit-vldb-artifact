from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_B2/trace_fixed"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_B2/trace_fixed/proxy_log.json'})

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
"B2 surrogate (fixed): TP-replica gradients all-reduced across rank, so\nembed.weight grad checksum is identical across simulated DP ranks. Reference\nrun for TrainCheck.\n\nWe don't have real distributed; simulate two ranks via two model copies with\ngradients explicitly synced post-backward.\n"
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

def main():
    annotate_stage('init')
    torch.manual_seed(0)
    rank0 = nn.Sequential(nn.Embedding(100, 8), nn.Linear(8, 4))
    rank1 = nn.Sequential(nn.Embedding(100, 8), nn.Linear(8, 4))
    rank1.load_state_dict(rank0.state_dict())
    opt0 = optim.AdamW(rank0.parameters(), lr=0.001)
    opt1 = optim.AdamW(rank1.parameters(), lr=0.001)
    for _ in range(8):
        x = torch.randint(0, 100, (2, 4))
        opt0.zero_grad()
        opt1.zero_grad()
        rank0(x).pow(2).sum().backward()
        rank1(x).pow(2).sum().backward()
        annotate_stage('testing')
        with torch.no_grad():
            for p0, p1 in zip(rank0.parameters(), rank1.parameters()):
                if p0.grad is not None and p1.grad is not None:
                    avg = (p0.grad + p1.grad) / 2
                    p0.grad.copy_(avg)
                    p1.grad.copy_(avg)
        annotate_stage('training')
        opt0.step()
        annotate_stage('training')
        opt1.step()
if __name__ == '__main__':
    main()