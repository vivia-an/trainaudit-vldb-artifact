"""Per-(pattern × framework) seed file inventory using pattern_hints.md
search hints. Each cell picks up to N representative files.

Output: seed_files.json  {framework: [{pattern, files, hint_pattern}]}
"""
import json, re
from pathlib import Path

REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
FRAMEWORKS = {
    "megatron":  REPO / "exp/frameworks/Megatron-LM",
    "deepspeed": REPO / "exp/frameworks/DeepSpeed",
    "olmo":      REPO / "exp/frameworks/OLMo",
    "olmo_core": REPO / "exp/frameworks/OLMo-core",
}

# 16 patterns × per-framework search regex (drawn from pattern_hints.md
# and the 8 P1-P8 patterns implicitly covered by existing T0/T1 rules).
PATTERNS = {
    "P1_dtype":        r"\.dtype|to\(torch\.|cast|amp\.|float16|bfloat16",
    "P2_scaling":      r"loss_scale|amp_scale|grad_scaler|max_grad_norm",
    "P3_cross_rank":   r"all_reduce|all_gather|reduce_scatter|broadcast",
    "P4_invocation":   r"def forward|@hook|forward_post|backward_post",
    "P5_state_restore":r"checkpoint|load_state_dict|save_state_dict",
    "P6_structural":   r"build|register_buffer|register_parameter",
    "P7_step_progress":r"global_step|trainer_step|optim\.step\b|state\[.step.\]",
    "P8_topology":     r"tensor_model_parallel|pipeline_model_parallel|ep_size|expert_parallel",
    "P9_init":         r"reset_parameters|kaiming_|trunc_normal_|mitchell_init|init_method",
    "P10_config_couple":r"zero_stage|mup_base_width|use_fsdp|sequence_parallel\s*=",
    "P11_pos_enc":     r"apply_rotary_pos_emb|cu_doc_lens|position_ids|RotaryEmb",
    "P12_alg_variant": r"flash_attn|fused_norm|use_(?:fused|flash)",
    "P13_aliasing":    r"data_ptr|\.view\(|share_memory_|register_buffer.*persistent=False",
    "P14_sharded":     r"mp_rank_|tp_rank.*save|partition_uneven",
    "P15_counter":     r"int32|torch\.int|sample_idx|step_count",
    "P16_loss_norm":   r"num_micro_batches|aux_loss_coeff|reduction\s*=\s*['\"]mean['\"]",
}


def find_seed_files(framework_root: Path, regex: str, max_files: int = 2):
    """Walk the framework root for .py files whose content matches regex,
    rank by # matches, return top-N file paths (relative to framework root)."""
    rx = re.compile(regex)
    hits = []
    for f in framework_root.rglob("*.py"):
        # Skip tests, vendor dirs
        if any(part in {"tests", "test", "build", "dist", "__pycache__",
                          "vendored", "third_party", "examples", "scripts"}
                for part in f.parts):
            continue
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        n = len(rx.findall(text))
        if n:
            hits.append((n, f))
    hits.sort(key=lambda t: -t[0])
    return [str(p.relative_to(framework_root)) for _, p in hits[:max_files]]


def main():
    out = {fw: [] for fw in FRAMEWORKS}
    for fw, root in FRAMEWORKS.items():
        if not root.exists():
            print(f"WARN missing {root}")
            continue
        for pid, rx in PATTERNS.items():
            files = find_seed_files(root, rx, max_files=2)
            out[fw].append({"pattern": pid, "regex": rx, "files": files})
            print(f"  {fw:10s} {pid:18s} n_files={len(files):2d} "
                  f"first={files[0] if files else '-'}")
    out_path = Path(__file__).parent / "seed_files.json"
    out_path.write_text(json.dumps(out, indent=2))
    n_total = sum(len(c["files"]) for fw_entries in out.values() for c in fw_entries)
    print(f"\nSaved {out_path} (total seed file refs = {n_total})")


if __name__ == "__main__":
    main()
