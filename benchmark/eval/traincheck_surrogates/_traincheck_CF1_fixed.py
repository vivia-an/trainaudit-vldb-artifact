from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_CF1/trace_fixed"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_CF1/trace_fixed/proxy_log.json'})

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
'CF1 surrogate (fixed): MoE aux-loss tracker guarded by torch.is_grad_enabled().\n\nRecompute pass under activation checkpointing runs without grad-enabled, so the guard\nsuppresses the second tracker call. calls_per_step == 1 invariant holds.\n'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

class AuxLossTracker:

    def __init__(self):
        self.calls_per_step = 0
        self.total = 0.0

    def save_to_aux_losses_tracker(self, value):
        self.calls_per_step += 1
        self.total += float(value)

    def end_step(self):
        c = self.calls_per_step
        self.calls_per_step = 0
        return c

class MoELayer(nn.Module):

    def __init__(self, tracker):
        super().__init__()
        self.gate = nn.Linear(8, 4)
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        self.tracker = tracker

    def forward(self, x):
        scores = self.gate(x).softmax(dim=-1)
        aux = (scores * scores.log().clamp(min=-10)).sum()
        if torch.is_grad_enabled():
            self.tracker.save_to_aux_losses_tracker(aux.item())
        weighted = sum((s.unsqueeze(-1) * e(x) for s, e in zip(scores.unbind(-1), self.experts)))
        return weighted

def main():
    annotate_stage('init')
    torch.manual_seed(0)
    tracker = AuxLossTracker()
    moe = MoELayer(tracker)
    model = moe
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    final_aux = 0.0
    for step in range(16):
        x = torch.randn(2, 8, requires_grad=True)
        out_first = moe(x)
        annotate_stage('testing')
        with torch.no_grad():
            _ = moe(x)
        loss = out_first.pow(2).sum()
        loss.backward()
        calls = tracker.end_step()
        if step == 0:
            print(f'[CF1_fixed] step 0 calls_per_step={calls} (expected 1)')
        final_aux = tracker.total
    print(f'[CF1_fixed] final aux_loss accumulator = {final_aux:.4f}')
    return final_aux
if __name__ == '__main__':
    main()