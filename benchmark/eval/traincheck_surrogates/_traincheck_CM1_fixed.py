from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_CM1/trace_fixed"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_CM1/trace_fixed/proxy_log.json'})

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
'CM1 surrogate (fixed): metric collective uses max_global; all ranks see same value.\n'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

def fake_collective_max(local_values, reduce_op='max_global'):
    if reduce_op == 'max_local':
        return local_values
    g = max(local_values)
    return [g] * len(local_values)

def main():
    annotate_stage('init')
    torch.manual_seed(0)
    import torch.nn as _nn
    from traincheck.instrumentor.tracer import Instrumentor
    Instrumentor(_nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
    model = _nn.Linear(1, 1)
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    n_ranks = 4
    n_steps = 32
    metric_log_per_rank = [[] for _ in range(n_ranks)]
    for step in range(n_steps):
        local_tput = [1000.0 + 5.0 * torch.randn(()).item() + 2.0 * r for r in range(n_ranks)]
        reduced = fake_collective_max(local_tput, reduce_op='max_global')
        for r in range(n_ranks):
            metric_log_per_rank[r].append(reduced[r])
    rank0_avg = sum(metric_log_per_rank[0]) / n_steps
    rank3_avg = sum(metric_log_per_rank[3]) / n_steps
    print(f'[CM1_fixed] rank0 avg tput = {rank0_avg:.2f}')
    print(f'[CM1_fixed] rank3 avg tput = {rank3_avg:.2f}')
    print(f'[CM1_fixed] rank-disagreement = {abs(rank3_avg - rank0_avg):.2f} (expected 0)')
    return (rank0_avg, rank3_avg)
if __name__ == '__main__':
    main()