from traincheck import annotate_stage
import os
os.environ.setdefault('ML_DAIKON_OUTPUT_DIR', "/tmp/tc_CM1/trace_buggy")
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
'CM1 surrogate (buggy): non-zero ranks see partially-reduced metric (max_local).'
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()


def fake_collective_max(local_values, reduce_op="max_global"):
    if reduce_op == "max_local":
        return local_values
    g = max(local_values)
    return [g] * len(local_values)


def main():
    annotate_stage('init')
    torch.manual_seed(0)
    n_ranks = 4
    n_steps = 32
    metric_log_per_rank = [[] for _ in range(n_ranks)]
    annotate_stage('testing')
    for step in range(n_steps):
        local_tput = [1000.0 + 5.0 * torch.randn(()).item() + 2.0 * r for r in range(n_ranks)]
        # buggy: max_local
        reduced = fake_collective_max(local_tput, reduce_op="max_local")
        for r in range(n_ranks):
            metric_log_per_rank[r].append(reduced[r])
        annotate_stage('training')


if __name__ == '__main__':
    main()
