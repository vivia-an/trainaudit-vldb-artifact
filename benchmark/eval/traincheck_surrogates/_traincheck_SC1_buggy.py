from traincheck import annotate_stage
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/tmp/tc_SC1/trace_buggy"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
general_config.MODEL_TRACKER_STYLE = 'proxy'
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/tmp/tc_SC1/trace_buggy/proxy_log.json'})

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
'SC1 surrogate (buggy): TP=2 checkpoint.save only writes rank-0 file.\n\nTriggers P14 Sharded State Completeness: under TP=N, save must produce a file\ncovering every TP rank in [0, N). Buggy short-circuits to rank 0 only.\n'
import os
import shutil
import tempfile

def fake_save(model_state, save_dir, tp_size, mode):
    os.makedirs(save_dir, exist_ok=True)
    if mode == 'buggy':
        path = os.path.join(save_dir, 'mp_rank_00_model_states.pt')
        with open(path, 'w') as f:
            f.write('rank-0 state')
    else:
        for r in range(tp_size):
            path = os.path.join(save_dir, f'mp_rank_{r:02d}_model_states.pt')
            with open(path, 'w') as f:
                f.write(f'rank-{r} state')

def main():
    annotate_stage('init')
    import torch.nn as _nn
    from traincheck.instrumentor.tracer import Instrumentor
    Instrumentor(_nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
    model = _nn.Linear(1, 1)
    model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
    tp_size = 2
    save_dir = tempfile.mkdtemp(prefix='sc1_buggy_')
    fake_save({'weight': 'stub'}, save_dir, tp_size, mode='buggy')
    saved_files = sorted(os.listdir(save_dir))
    expected_files = sorted([f'mp_rank_{r:02d}_model_states.pt' for r in range(tp_size)])
    missing = set(expected_files) - set(saved_files)
    print(f'[SC1_buggy] tp_size={tp_size}')
    print(f'[SC1_buggy] saved   : {saved_files}')
    print(f'[SC1_buggy] expected: {expected_files}')
    print(f'[SC1_buggy] missing : {sorted(missing)}  (P14 violation: {len(missing)} ranks not saved)')
    shutil.rmtree(save_dir)
    return len(missing)
if __name__ == '__main__':
    main()