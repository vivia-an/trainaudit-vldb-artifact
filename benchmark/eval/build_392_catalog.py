"""
Build unified 392-bug catalog from 128 pool + 295 pool + 1 orphan.

Steps:
1. Create benchmark/bugs/<id>/config.json for 96 bugs unique to 128 pool
2. Mechanically map 295 pool category field to the 13-class taxonomy where possible
3. Flag remaining bugs that need LLM re-classification

Outputs:
- benchmark/bugs/<id>/config.json (96 new dirs from 128-only)
- benchmark/eval/category_alias_map.json (mechanical mapping rules used)
- benchmark/eval/category_remaining.json (bugs still needing LLM)
- benchmark/eval/category_resolved.json (per-bug final category after rules)
"""
import json
from pathlib import Path
from collections import Counter

CAT13 = {'numerical','checkpoint','gradient_sync','communication','control_flow',
         'sharding','dtype','moe','optimizer_state','loss_computation','data_loading',
         'offload','lr_schedule'}

# Mechanical alias map. Built from inspecting all 57 borderline values in 295 pool.
ALIAS_MAP = {
    # optimizer family
    'optimizer': 'optimizer_state',
    'zero_optimizer': 'optimizer_state',
    'zero': 'optimizer_state',
    # loss family
    'loss': 'loss_computation',
    'loss_computation/moe': 'loss_computation',
    'loss_scaling': 'numerical',     # loss scaling is a numerical concern
    # checkpoint family
    'checkpointing': 'checkpoint',
    'checkpoint/retention': 'checkpoint',
    # data family
    'data_processing': 'data_loading',
    'data': 'data_loading',
    'data_loader/sharding': 'data_loading',
    'data_loader/epoch_transition': 'data_loading',
    'data_loader/scheduling': 'data_loading',
    'tokenizer/eos_token': 'data_loading',
    # dtype family
    'mixed_precision': 'dtype',
    'precision': 'dtype',
    'communication_dtype': 'dtype',
    'quantization': 'dtype',
    # MoE family
    'routing': 'moe',
    'moe_router_init': 'moe',
    'moe_parallel_grouping': 'moe',
    # lr_schedule family
    'lr_scheduler': 'lr_schedule',
    'scheduler': 'lr_schedule',
    'scheduler/learning_rate': 'lr_schedule',
    # gradient_sync family
    'gradient_clipping': 'gradient_sync',
    'gradient_scaling': 'gradient_sync',
    'gradient_corruption': 'gradient_sync',
    'gradient_reduction': 'gradient_sync',
    'gradient_accumulation': 'gradient_sync',
    'tensor_parallel_grad': 'gradient_sync',
    'pipeline_parallel_sync': 'gradient_sync',
    'pipeline': 'gradient_sync',
    # communication family
    'sequence_parallel': 'communication',
    'context_parallel': 'communication',
    # control_flow family (init/config/freeze)
    'freeze': 'control_flow',
    'config_parsing': 'control_flow',
    'config': 'control_flow',
    'config/model_size': 'control_flow',
    'configuration_validation': 'control_flow',
    'state_mutation': 'control_flow',
    'parallelism_initialization': 'control_flow',
    'initialization': 'control_flow',
    'model_init/fsdp': 'control_flow',
    'model_init/weights': 'control_flow',
    'model_init/layer_norm_config': 'control_flow',
    'model_init/embedding': 'control_flow',
    'model_init/mup': 'control_flow',
    # metric/tracking → control_flow (counter / state tracking bugs)
    'metric_tracking': 'control_flow',
    'metrics': 'control_flow',
    # numerical family (model arch / compute)
    'attention': 'numerical',
    'normalization': 'numerical',
    'rope': 'numerical',
    'positional_encoding': 'numerical',
    'residual_connection': 'numerical',
    'te_integration': 'numerical',
    'cuda_graph': 'numerical',
}

ROOT = Path('/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025')


def step1_build_128_only_configs():
    """For 96 bugs unique to 128 pool, create benchmark/bugs/<id>/config.json"""
    pool_overlap = json.loads((ROOT/'benchmark/eval/pool_overlap.json').read_text())
    only128 = pool_overlap['only128_ids']
    print(f"Step 1: build configs for {len(only128)} 128-only bugs")

    # Load 128 pool data
    pool128 = {}
    for fw_file in ['megatron_silent_errors.json','deepspeed_silent_errors.json','olmo_silent_errors.json']:
        for b in json.loads((ROOT/'exp/data'/fw_file).read_text()):
            pool128[b['id']] = b

    # Framework inference: M-* → megatron-lm, D-* → deepspeed, O-* → olmo or olmo-core (by repo)
    fw_prefix = {'M':'megatron-lm', 'D':'deepspeed'}

    created = 0
    olmo_split = {'olmo':0, 'olmo-core':0}
    for bid in only128:
        src = pool128[bid]
        # Figure framework
        if bid.startswith(('M-','D-')):
            framework = fw_prefix[bid[0]]
        elif bid.startswith('O-'):
            repo = src.get('repo','')
            if 'OLMo-core' in repo:
                framework = 'olmo-core'
            else:
                framework = 'olmo'
            olmo_split[framework] += 1
        else:
            framework = 'unknown'

        # Build config.json with merged schema (295 fields + 128 fine fields)
        cfg = {
            'bug_id': bid,
            'source_pool': '128_only',
            'framework': framework,
            'repo': src.get('repo',''),
            'title': src.get('title',''),
            'description': src.get('description',''),
            'issue_url': src.get('url') or src.get('issue_or_pr',''),
            'category': src.get('category'),  # 128 pool categories are all clean
            'severity': src.get('severity','unset'),
            # 128-pool fine fields (preserved)
            'parallel_dimension': src.get('parallel_dimension'),
            'detection_signal': src.get('detection_signal'),
            'required_trace_fields': src.get('required_trace_fields',[]),
            'check_stage': src.get('check_stage'),
            'invariant_type': src.get('invariant_type'),
            # 295-pool style fields not in 128 pool (filled with placeholders)
            'buggy_commit': None,
            'fixed_commit': None,
            'expected_output': None,
            'gpu_needed': None,
            'trigger_conditions': [],
            'root_cause': src.get('description',''),  # use description as root_cause
            'invariant': None,
            'detection_method': src.get('detection_signal'),
            'reproduction_status': '<unset>',  # 128-only bugs not yet reproduced
            'has_detect_py': False,
            'has_reproduce_sh': False,
            'has_trainaudit_driver': False,
            'missing_fields': ['buggy_commit','fixed_commit','expected_output','gpu_needed','invariant'],
        }

        bug_dir = ROOT/'benchmark/bugs'/bid
        bug_dir.mkdir(parents=True, exist_ok=True)
        (bug_dir/'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        created += 1

    print(f"  Created {created} config.json")
    print(f"  OLMo split: olmo={olmo_split['olmo']}, olmo-core={olmo_split['olmo-core']}")
    return created


def step2_map_295_categories():
    """Apply ALIAS_MAP to 295 pool. Output:
    - category_resolved.json: per-bug final category
    - category_remaining.json: bugs still in <missing> or unmapped → need LLM
    """
    print(f"\nStep 2: map 295 pool category to 13-class")
    manifest = json.loads((ROOT/'benchmark/eval/manifest.json').read_text())
    resolved = []
    remaining = []
    stats = Counter()

    for b in manifest:
        bid = b['bug_id']
        old_cat = b.get('category', '<missing>')

        if old_cat in CAT13:
            new_cat = old_cat
            method = 'already_in_13_class'
        elif old_cat in ALIAS_MAP:
            new_cat = ALIAS_MAP[old_cat]
            method = f'alias_map ({old_cat} → {new_cat})'
        else:
            # Unmapped: <missing>, or rare categories not in alias map
            new_cat = None
            method = 'NEEDS_LLM'

        rec = {
            'bug_id': bid,
            'framework': b.get('framework'),
            'old_category': old_cat,
            'new_category': new_cat,
            'method': method,
            'title': b.get('title',''),
            'root_cause': b.get('root_cause',''),
            'invariant': b.get('invariant',''),
        }

        if new_cat:
            resolved.append(rec)
            stats[method.split(' (')[0]] += 1
        else:
            remaining.append(rec)
            stats['NEEDS_LLM'] += 1

    (ROOT/'benchmark/eval/category_resolved.json').write_text(
        json.dumps({'count': len(resolved), 'records': resolved}, indent=2, ensure_ascii=False)
    )
    (ROOT/'benchmark/eval/category_remaining.json').write_text(
        json.dumps({'count': len(remaining), 'records': remaining}, indent=2, ensure_ascii=False)
    )

    print(f"  Method breakdown:")
    for m, n in stats.most_common():
        print(f"    {m}: {n}")
    print(f"  Resolved (in 13 class): {len(resolved)}/295")
    print(f"  NEEDS_LLM: {len(remaining)}")

    # Also save the alias map for reference
    (ROOT/'benchmark/eval/category_alias_map.json').write_text(
        json.dumps({'version':'v1','13_classes':sorted(CAT13),'aliases':ALIAS_MAP}, indent=2, ensure_ascii=False)
    )
    return resolved, remaining


if __name__ == '__main__':
    step1_build_128_only_configs()
    resolved, remaining = step2_map_295_categories()
    print(f"\n=== Done ===")
    print(f"Step 1: 96 configs created in benchmark/bugs/")
    print(f"Step 2: {len(resolved)}/295 resolved, {len(remaining)} need LLM")
