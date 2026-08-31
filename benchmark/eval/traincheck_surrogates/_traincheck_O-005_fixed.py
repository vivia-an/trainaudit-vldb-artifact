from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_O-005/trace_fixed"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_O-005/trace_fixed/proxy_log.json'})

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
'O-005 surrogate (fixed): torch.utils.checkpoint with preserve_rng_state=True\nkeeps Dropout state aligned across forward + recompute.\n\nReference run for TrainCheck.\n'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.utils import checkpoint as cp
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(cp, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

class BlockWithDropout(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(self.linear(x))

class TopModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 8)
        self.blk = BlockWithDropout()
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        h = self.embed(x)
        h = cp.checkpoint(self.blk, h, use_reentrant=False, preserve_rng_state=True)
        return self.head(h)

def main():
    annotate_stage('init')
    torch.manual_seed(0)
    model = TopModel()
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    opt = optim.AdamW(model.parameters(), lr=0.001)
    for step in range(4):
        opt.zero_grad()
        x = torch.randn(2, 8, requires_grad=True)
        model(x).pow(2).sum().backward()
        annotate_stage('training')
        opt.step()
if __name__ == '__main__':
    main()