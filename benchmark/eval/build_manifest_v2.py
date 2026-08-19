"""
Step 3: build unified manifest_v2.json (392 bugs).

Sources:
- 96 bugs unique to 128 pool (just created in benchmark/bugs/<id>/config.json)
- 295 bugs in 295 pool (benchmark/eval/manifest.json + category_resolved.json)
- 1 orphan: M-NEW-MUON-MTP (already has its own config.json, not in manifest.json)
- Excluded: M-NEW-2 (no config.json, deemed incomplete in Phase 0)

For overlap bugs (32): we use 295-pool metadata as the base, but verify category
matches between 128 and 295 pool annotations. Record category_origin.

Output: benchmark/eval/manifest_v2.json
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path('/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025')

CAT13 = {'numerical','checkpoint','gradient_sync','communication','control_flow',
         'sharding','dtype','moe','optimizer_state','loss_computation','data_loading',
         'offload','lr_schedule'}


def main():
    pool_overlap = json.loads((ROOT/'benchmark/eval/pool_overlap.json').read_text())
    overlap = set(pool_overlap['overlap_ids'])
    only128 = set(pool_overlap['only128_ids'])
    only295 = set(pool_overlap['only295_ids'])

    # Load 128 pool data (for category cross-check on overlap)
    pool128 = {}
    for fw_file in ['megatron_silent_errors.json','deepspeed_silent_errors.json','olmo_silent_errors.json']:
        for b in json.loads((ROOT/'exp/data'/fw_file).read_text()):
            pool128[b['id']] = b

    # Load 295 pool resolved categories
    cat_resolved = json.loads((ROOT/'benchmark/eval/category_resolved.json').read_text())
    cat_map = {r['bug_id']: r for r in cat_resolved['records']}

    # Load 295 pool full metadata
    manifest295 = {b['bug_id']: b for b in json.loads((ROOT/'benchmark/eval/manifest.json').read_text())}

    merged = []
    cat_disagreement = []  # 128 vs 295 category mismatch on overlap

    # Pass 1: 128-only bugs (from benchmark/bugs/<id>/config.json which we created)
    for bid in sorted(only128):
        cfg_path = ROOT/'benchmark/bugs'/bid/'config.json'
        cfg = json.loads(cfg_path.read_text())
        rec = {
            'bug_id': bid,
            'source_pool': '128_only',
            'framework': cfg['framework'],
            'repo': cfg['repo'],
            'title': cfg['title'],
            'issue_url': cfg.get('issue_url',''),
            'category': cfg['category'],
            'category_origin': '128_pool_curated',
            'parallel_dimension': cfg.get('parallel_dimension'),
            'severity': cfg.get('severity'),
            'description': cfg.get('description',''),
            'root_cause': cfg.get('root_cause',''),
            'invariant': cfg.get('invariant'),
            # 128-pool fine fields
            'invariant_type': cfg.get('invariant_type'),
            'required_trace_fields': cfg.get('required_trace_fields',[]),
            'check_stage': cfg.get('check_stage'),
            'detection_signal': cfg.get('detection_signal'),
            # 295-pool style fields (placeholders since we don't have them)
            'buggy_commit': None,
            'fixed_commit': None,
            'expected_output': None,
            'gpu_needed': None,
            'trigger_conditions': [],
            'detection_method': cfg.get('detection_signal'),
            'reproduction_status': '<unset>',
            'has_detect_py': False,
            'has_reproduce_sh': False,
            'has_trainaudit_driver': False,
        }
        merged.append(rec)

    # Pass 2: 295 pool bugs (overlap ∪ only295 = 295)
    for bid, b in manifest295.items():
        is_overlap = bid in overlap
        cat_rec = cat_map[bid]
        new_cat = cat_rec['new_category']
        cat_method = cat_rec['method']

        # On overlap, prefer 128 pool category (curated). Record disagreement.
        if is_overlap and bid in pool128:
            cat_128 = pool128[bid]['category']
            if cat_128 != new_cat:
                cat_disagreement.append({
                    'bug_id': bid,
                    'cat_128_pool': cat_128,
                    'cat_295_resolved': new_cat,
                    'cat_295_old': cat_rec['old_category'],
                })
                # Override with 128 pool
                new_cat = cat_128
                cat_method = '128_pool_override (was: ' + cat_method + ')'
            else:
                cat_method += ' (matches 128 pool)'

        rec = {
            'bug_id': bid,
            'source_pool': 'both' if is_overlap else '295_only',
            'framework': b.get('framework'),
            'repo': pool128.get(bid,{}).get('repo','') if is_overlap else b.get('framework',''),
            'title': b.get('title',''),
            'issue_url': b.get('issue_url',''),
            'category': new_cat,
            'category_origin': cat_method,
            'parallel_dimension': pool128.get(bid,{}).get('parallel_dimension') if is_overlap else None,
            'severity': b.get('severity'),
            'description': pool128.get(bid,{}).get('description','') if is_overlap else '',
            'root_cause': b.get('root_cause',''),
            'invariant': b.get('invariant'),
            # 128-pool fine fields (only if overlap)
            'invariant_type': pool128.get(bid,{}).get('invariant_type') if is_overlap else None,
            'required_trace_fields': pool128.get(bid,{}).get('required_trace_fields',[]) if is_overlap else [],
            'check_stage': pool128.get(bid,{}).get('check_stage') if is_overlap else None,
            'detection_signal': pool128.get(bid,{}).get('detection_signal') if is_overlap else None,
            # 295-pool fields
            'buggy_commit': b.get('buggy_commit'),
            'fixed_commit': b.get('fixed_commit'),
            'expected_output': b.get('expected_output'),
            'gpu_needed': b.get('gpu_needed'),
            'trigger_conditions': b.get('trigger_conditions',[]),
            'detection_method': b.get('detection_method'),
            'reproduction_status': b.get('reproduction_status','<unset>'),
            'has_detect_py': b.get('has_detect_py', False),
            'has_reproduce_sh': b.get('has_reproduce_sh', False),
            'has_trainaudit_driver': b.get('has_trainaudit_driver', False),
        }
        merged.append(rec)

    # Pass 3: orphan
    orphan_path = ROOT/'benchmark/bugs/M-NEW-MUON-MTP/config.json'
    orphan_cfg = json.loads(orphan_path.read_text())
    orphan_cat = orphan_cfg.get('category','optimizer_state')
    if orphan_cat not in CAT13:
        # alias check
        from build_392_catalog import ALIAS_MAP
        orphan_cat = ALIAS_MAP.get(orphan_cat, 'optimizer_state')
    rec = {
        'bug_id': 'M-NEW-MUON-MTP',
        'source_pool': '295_orphan',
        'framework': orphan_cfg.get('framework','megatron-lm'),
        'repo': orphan_cfg.get('repo','NVIDIA/Megatron-LM'),
        'title': orphan_cfg.get('title',''),
        'issue_url': orphan_cfg.get('issue_url',''),
        'category': orphan_cat,
        'category_origin': 'orphan_config_json',
        'parallel_dimension': orphan_cfg.get('parallel_dimensions',['PP'])[0] if orphan_cfg.get('parallel_dimensions') else None,
        'severity': orphan_cfg.get('severity'),
        'description': '',
        'root_cause': orphan_cfg.get('root_cause',''),
        'invariant': orphan_cfg.get('invariant'),
        'invariant_type': None,
        'required_trace_fields': [],
        'check_stage': None,
        'detection_signal': None,
        'buggy_commit': orphan_cfg.get('buggy_commit'),
        'fixed_commit': orphan_cfg.get('fixed_commit'),
        'expected_output': orphan_cfg.get('expected_output'),
        'gpu_needed': orphan_cfg.get('gpu_needed'),
        'trigger_conditions': orphan_cfg.get('trigger_conditions',[]),
        'detection_method': orphan_cfg.get('detection_method'),
        'reproduction_status': orphan_cfg.get('reproduction_status','<unset>'),
        'has_detect_py': True,
        'has_reproduce_sh': True,
        'has_trainaudit_driver': False,
    }
    merged.append(rec)

    # Sanity
    assert len(merged) == 392, f"Expected 392, got {len(merged)}"
    by_pool = Counter(r['source_pool'] for r in merged)
    print(f"Total: {len(merged)} (expect 392)")
    print(f"By pool: {dict(by_pool)}")

    # Validate every category in 13 classes
    bad_cats = [r for r in merged if r['category'] not in CAT13]
    assert not bad_cats, f"Bad cats: {[(r['bug_id'], r['category']) for r in bad_cats]}"
    print(f"All 392 categories ∈ 13-class taxonomy ✓")

    # Disagreements report
    print(f"\nCategory disagreements (128 vs 295 on overlap): {len(cat_disagreement)}")
    for d in cat_disagreement[:10]:
        print(f"  {d['bug_id']}: 128={d['cat_128_pool']} vs 295={d['cat_295_resolved']} (was: {d['cat_295_old']})")
    if len(cat_disagreement) > 10:
        print(f"  ... and {len(cat_disagreement)-10} more")

    # Save
    out = {
        'meta': {
            'version': 'v2',
            'date': '2026-05-10',
            'total': len(merged),
            'by_source_pool': dict(by_pool),
            'category_disagreements_overlap': len(cat_disagreement),
        },
        'category_disagreements': cat_disagreement,
        'bugs': merged,
    }
    (ROOT/'benchmark/eval/manifest_v2.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote benchmark/eval/manifest_v2.json")
    return out


if __name__ == '__main__':
    main()
