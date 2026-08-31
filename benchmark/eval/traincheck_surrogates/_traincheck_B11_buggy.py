from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_B11/trace_buggy"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_B11/trace_buggy/proxy_log.json'})

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
'B11 surrogate (buggy): grad clipping is replaced with a no-op that only\nreturns the norm without actually rescaling gradients.\n\nMirrors the DeepSpeed `ds_utils.clip_grad_norm_` regression at commit 005afe12\nwhere the routine returned the L2 norm but never multiplied parameters by the\nclip coefficient. After this "clip", grads remain unbounded.\n'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn.utils as nn_utils
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn_utils, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

def buggy_clip(parameters, max_norm, norm_type=2.0, **kwargs):
    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float(), norm_type) for p in params]), norm_type)

def main():
    annotate_stage('init')
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    opt = optim.AdamW(model.parameters(), lr=0.001)
    max_norm = 0.1
    nn_utils.clip_grad_norm_ = buggy_clip
    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        loss = model(x).pow(2).sum() * 1000000.0
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        annotate_stage('training')
        opt.step()
if __name__ == '__main__':
    main()