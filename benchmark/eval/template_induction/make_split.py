"""Frozen dataset split for the template induction experiment.

Seed=128 / Development=186 / LockedTest=78 over the 392-bug pool
(manifest_v2.json). Constraints per template_coding_protocol.md §5.
RNG seed = 20260716.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
EVAL = HERE.parent

RNG = random.Random(20260716)

WHITELIST = ["bug_id", "framework", "repo", "title", "description", "root_cause",
             "category", "parallel_dimension", "severity", "trigger_conditions",
             "issue_url"]


def main():
    bugs = json.load(open(EVAL / "manifest_v2.json"))["bugs"]
    by_id = {b["bug_id"]: b for b in bugs}
    assert len(bugs) == 392

    rs = json.load(open(EVAL / "real_sdc" / "real_sdc_manifest.json"))
    real_se_cases = rs["cases_confirmed_real"] + rs.get("cases_boundary", [])
    real_se_ids = []
    for c in real_se_cases:
        bid = c["case_id"] if c["case_id"] in by_id else c["original_bug_id"]
        assert bid in by_id, bid
        real_se_ids.append(bid)
    assert len(real_se_ids) == 18

    # Leakage groups: same normalized issue_url -> same group.
    groups = {}
    url_group = {}
    for b in bugs:
        url = (b.get("issue_url") or "").rstrip("/")
        if url and url in url_group:
            gid = url_group[url]
        else:
            gid = b["bug_id"]
            if url:
                url_group[url] = gid
        groups.setdefault(gid, []).append(b["bug_id"])

    gid_of = {bid: gid for gid, members in groups.items() for bid in members}

    # 1) Locked test: Real-SE groups first.
    test = set()
    for bid in real_se_ids:
        test.update(groups[gid_of[bid]])
    forced_n = len(test)

    # Top up to 78 with stratified sampling by framework x category,
    # proportional to the remaining pool, at group granularity.
    remaining_groups = [g for g in groups if not any(m in test for m in groups[g])]
    need = 78 - forced_n
    strata = defaultdict(list)
    for g in remaining_groups:
        rep = by_id[groups[g][0]]
        strata[(rep["framework"], rep["category"])].append(g)
    for gs in strata.values():
        gs.sort()
        RNG.shuffle(gs)
    total_remaining = sum(len(groups[g]) for g in remaining_groups)
    # Largest-remainder allocation of `need` cases across strata.
    quotas = {}
    fracs = []
    used = 0
    for key, gs in sorted(strata.items()):
        n_cases = sum(len(groups[g]) for g in gs)
        exact = need * n_cases / total_remaining
        quotas[key] = int(exact)
        used += int(exact)
        fracs.append((exact - int(exact), key))
    fracs.sort(reverse=True)
    for _, key in fracs[: need - used]:
        quotas[key] += 1

    for key, gs in sorted(strata.items()):
        want = quotas[key]
        got = 0
        while gs and got < want:
            g = gs.pop()
            members = groups[g]
            if len(test) + len(members) > 78:
                continue
            test.update(members)
            got += len(members)
    # Fill any shortfall (quota rounding vs group sizes) from leftover groups.
    leftover = [g for g in groups
                if not any(m in test for m in groups[g])]
    leftover.sort()
    RNG.shuffle(leftover)
    for g in leftover:
        if len(test) >= 78:
            break
        if len(test) + len(groups[g]) <= 78:
            test.update(groups[g])
    assert len(test) == 78, len(test)

    # 2) Seed: original 128-pool membership minus test, refilled stratified.
    orig128 = [b["bug_id"] for b in bugs if b["source_pool"] in ("128_only", "both")]
    seed = set()
    for bid in orig128:
        if gid_of[bid] not in {gid_of[t] for t in test}:
            seed.update(groups[gid_of[bid]])
    pool_rest = [g for g in groups
                 if not any(m in test or m in seed for m in groups[g])]
    strata2 = defaultdict(list)
    for g in pool_rest:
        rep = by_id[groups[g][0]]
        strata2[(rep["framework"], rep["category"])].append(g)
    for gs in strata2.values():
        gs.sort()
        RNG.shuffle(gs)
    need2 = 128 - len(seed)
    total2 = sum(len(groups[g]) for g in pool_rest)
    quotas2 = {}
    fracs2 = []
    used2 = 0
    for key, gs in sorted(strata2.items()):
        n_cases = sum(len(groups[g]) for g in gs)
        exact = need2 * n_cases / total2
        quotas2[key] = int(exact)
        used2 += int(exact)
        fracs2.append((exact - int(exact), key))
    fracs2.sort(reverse=True)
    for _, key in fracs2[: need2 - used2]:
        quotas2[key] += 1
    for key, gs in sorted(strata2.items()):
        want = quotas2[key]
        got = 0
        while gs and got < want:
            g = gs.pop()
            members = groups[g]
            if len(seed) + len(members) > 128:
                continue
            seed.update(members)
            got += len(members)
    leftover2 = [g for g in groups
                 if not any(m in test or m in seed for m in groups[g])]
    leftover2.sort()
    RNG.shuffle(leftover2)
    for g in leftover2:
        if len(seed) >= 128:
            break
        if len(seed) + len(groups[g]) <= 128:
            seed.update(groups[g])
    assert len(seed) == 128, len(seed)

    dev = set(by_id) - test - seed
    assert len(dev) == 186

    # Development stream order (frozen).
    dev_order = sorted(dev)
    RNG.shuffle(dev_order)
    batches = [dev_order[i:i + 25] for i in range(0, len(dev_order), 25)]

    split = {
        "meta": {
            "date": "2026-07-16",
            "rng_seed": 20260716,
            "pool": "manifest_v2.json (392 bugs)",
            "leakage_groups_multi": {g: m for g, m in groups.items() if len(m) > 1},
            "real_se_forced_test": sorted(real_se_ids),
            "forced_test_with_leakage_mates": forced_n,
            "note": "no temporal split: 0/392 cases carry fix_date",
        },
        "seed": sorted(seed),
        "development": sorted(dev),
        "development_stream_order": dev_order,
        "development_batches": batches,
        "test": sorted(test),
    }
    json.dump(split, open(HERE / "dataset_split.json", "w"),
              indent=1, ensure_ascii=False)

    def dist(ids):
        fw = Counter(by_id[i]["framework"] for i in ids)
        cat = Counter(by_id[i]["category"] for i in ids)
        return fw, cat

    for name, ids in [("seed", seed), ("dev", dev), ("test", test)]:
        fw, cat = dist(ids)
        print(f"{name} ({len(ids)}): {dict(fw)}")
        print(f"   cat: {dict(cat)}")

    # Whitelisted annotator inputs.
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)
    def dump_cases(ids, path):
        rows = [{k: by_id[i].get(k) for k in WHITELIST} for i in sorted(ids)]
        json.dump(rows, open(path, "w"), indent=1, ensure_ascii=False)
    dump_cases(seed, inputs / "seed_cases.json")
    dump_cases(dev, inputs / "development_cases.json")
    dump_cases(test, inputs / "test_cases.json")
    for i, batch in enumerate(batches, 1):
        dump_cases(batch, inputs / f"dev_batch_{i:02d}.json")
    print("inputs written:", len(list(inputs.glob('*.json'))), "files")


if __name__ == "__main__":
    main()
