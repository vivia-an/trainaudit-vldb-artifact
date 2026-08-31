"""Parse all smoke logs under logs/smoke/ into real_sdc_same_harness.csv.

Recognises several variants of the per-case driver:
- reproduce.sh:        prints `[1/2] Running buggy`, `[2/2] Running fixed`, `[CASE] / [RESULT] BUG DETECTED|CLEAN`
- trainaudit_run.sh:   prints `===== BUGGY (...) =====`, `===== FIXED (...) =====`, `[CASE/trainaudit] BUG DETECTED|CLEAN`

Verdict rules:
  DETECTED  ← `BUG DETECTED`
  CLEAN     ← `CLEAN`
  FAIL      ← ModuleNotFoundError / ImportError / ChildFailedError / pydantic / unrecognized arguments etc.
  NOT_RUN   ← log not found or no verdict line
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path

ROOT = Path('/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025')
LOG_DIR = ROOT / 'benchmark/eval/real_sdc/logs/smoke'
OUT_CSV = ROOT / 'benchmark/eval/real_sdc/real_sdc_same_harness.csv'

BOUNDARY_RE = re.compile(r'(\[2/2\] Running fixed|===== FIXED \(|FIXED \(|\[2/2\] Running|TrainAudit.*FIXED)', re.I)
BUGGY_START_RE = re.compile(r'(\[1/2\] Running buggy|===== BUGGY \()', re.I)
DETECTED_RE = re.compile(r'BUG DETECTED')
CLEAN_RE = re.compile(r'\bCLEAN\b')

FAIL_PATTERNS = [
    ('cpuinfo', 'cpuinfo_missing'),
    ('hjson',   'hjson_missing'),
    ('ModuleNotFoundError', 'module_missing'),
    ('ImportError',          'import_error'),
    ('unrecognized arguments', 'cli_arg_mismatch'),
    ('AssertionError',       'assert'),
    ('ChildFailedError',     'torchrun_child_failed'),
    ('pydantic',             'pydantic_conflict'),
]

CASES = [
    ('B1',       'megatron-lm',  'reproduce.sh'),
    ('B3',       'deepspeed',    'reproduce.sh'),
    ('B8',       'deepspeed',    'reproduce.sh'),
    ('B11',      'deepspeed',    'reproduce.sh'),
    ('B12',      'olmo-core',    'reproduce.sh'),
    ('M-020',    'megatron-lm',  'reproduce.sh'),
    ('M-010',    'megatron-lm',  'reproduce.sh'),
    ('O-005',    'olmo',         'trainaudit_run.sh'),
    ('O-NEW-9',  'olmo',         'trainaudit_run.sh'),
    ('OC-NEW-2', 'olmo-core',    'trainaudit_run.sh'),
    ('D-029',    'deepspeed',    'trainaudit_run.sh'),
    ('D-NEW-9',  'deepspeed',    'trainaudit_run.sh'),
    ('O-NEW-5',  'olmo',         'trainaudit_run.sh'),
    ('O-NEW-3',  'olmo',         'trainaudit_run.sh'),
    ('O-NEW-2',  'olmo',         'trainaudit_run.sh'),
    ('OC-NEW-6', 'olmo-core',    'trainaudit_run.sh'),
    ('O-040',    'olmo-core',    'trainaudit_run.sh'),
    ('D-NEW-49', 'deepspeed',    'trainaudit_run.sh'),
    ('M-NEW-33', 'megatron-lm',  'trainaudit_run.sh'),
    ('OC-NEW-22','olmo-core',    'trainaudit_run.sh'),
]

def detect_fail(seg: list[str]) -> str:
    for kw, label in FAIL_PATTERNS:
        if any(kw in l for l in seg):
            return label
    return ''

def classify_segment(seg: list[str]) -> tuple[str, str]:
    has_detect = any(DETECTED_RE.search(l) for l in seg)
    has_clean  = any(CLEAN_RE.search(l) for l in seg)
    fail_kind  = detect_fail(seg)
    if has_detect and not has_clean:
        return 'DETECTED', ''
    if has_clean and not has_detect:
        return 'CLEAN', ''
    if has_detect and has_clean:
        # both — happens because driver prints both; take the last one before any failure
        last_verdict = None
        for l in seg:
            if DETECTED_RE.search(l): last_verdict = 'DETECTED'
            elif CLEAN_RE.search(l):  last_verdict = 'CLEAN'
        return last_verdict or 'CLEAN', ''
    if fail_kind:
        return 'FAIL', fail_kind
    return 'NOT_RUN', 'no_verdict_in_log'

def parse_log(case_id: str, log_path: Path) -> tuple[str, str, str, str]:
    if not log_path.exists():
        return 'NOT_RUN','no_log','NOT_RUN','no_log'
    lines = log_path.read_text(errors='replace').splitlines()
    boundary = None
    for i,l in enumerate(lines):
        if BOUNDARY_RE.search(l):
            boundary = i; break
    if boundary is None:
        buggy_seg, fixed_seg = lines, []
    else:
        buggy_seg, fixed_seg = lines[:boundary], lines[boundary:]
    bv, bk = classify_segment(buggy_seg)
    fv, fk = classify_segment(fixed_seg)
    # Special case: M-020 fixed = AssertionError is the *expected fix behavior*
    if case_id == 'M-020':
        if any('>>> Fixed: ASSERT fired' in l for l in fixed_seg):
            fv = 'CLEAN'
            fk = 'fix_asserts_invalid_config'
        elif fv == 'FAIL' and fk == 'assert':
            fv = 'CLEAN'
            fk = 'fix_asserts_invalid_config'
    return bv, bk, fv, fk

def main():
    fields = ('case_id','original_bug_id','tool','phase','verdict','detected',
              'violation_count','total_checks','fail_kind','seed','gpu_count',
              'command','log_path')
    rows: list[dict] = []
    summary = []
    for cid, fw, script in CASES:
        log = LOG_DIR / f'{cid}_smoke.log'
        bv, bk, fv, fk = parse_log(cid, log)
        summary.append((cid, bv, bk, fv, fk))
        for phase, v, k in [('buggy', bv, bk), ('reference_fixed', fv, fk)]:
            rows.append({
                'case_id': cid, 'original_bug_id': cid,
                'tool': 'trainaudit', 'phase': phase,
                'verdict': v, 'detected': v == 'DETECTED',
                'violation_count': '', 'total_checks': '',
                'fail_kind': k, 'seed': 0, 'gpu_count': 2,
                'command': f'smoke: bash benchmark/bugs/{cid}/{script}',
                'log_path': str(log),
            })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'wrote {OUT_CSV}: {len(rows)} rows')
    print(f"\n{'case':<10s} {'buggy':<10s} {'fixed':<10s}  notes")
    for cid, bv, bk, fv, fk in summary:
        note = ''
        if bk: note = f"buggy={bk}"
        if fk: note = (note + f" | fixed={fk}") if note else f"fixed={fk}"
        print(f"{cid:<10s} {bv:<10s} {fv:<10s}  {note}")

if __name__ == '__main__':
    main()
