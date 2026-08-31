# Upstream evidence for the production discovery (§5.6, §1)

The paper's flagship out-of-pool result — a rule firing before the first optimizer step
on a 256×H800 job, "reported upstream, where maintainers confirmed and fixed this
previously unrecorded bug" — is currently stated without an identifier. It was written
that way for double-blind review. **VLDB is single-blind, so the identifier can now be
given**, which turns the strongest claim in the paper from unverifiable into checkable.

Verified upstream state (checked 2026-08-19):

| Item | State |
|---|---|
| `NVIDIA/Megatron-LM` issue **#4641** — "[BUG] Muon + PP + MTP: tied word_embeddings on MTP-only last PP stage is routed to Muon instead of Adam" | **closed**; labeled a bug and assigned to a maintainer; opened 2026-05-06 by `yezhengmao1` (co-author Zhengmao Ye) |
| `NVIDIA/Megatron-LM` PR **#4642** — fix for the same routing bug | **closed, not merged**; a duplicate fix (PR #5034) landed first |

Root cause as recorded upstream: the `is_embedding_or_output_parameter` tags were only
applied when `self.pre_process` was true, so on an MTP-only last stage
(`pre_process=False`) the tied parameter stayed untagged and was routed to Muon instead
of Adam — optimizer divergence with orphaned state at checkpoint time. That matches the
paper's description at `main.tex:1090–1100` exactly.

## Recommended wording

Cite the issue, and keep the fix attribution accurate: the maintainers confirmed the
report and the bug is fixed upstream, but the authors' own PR was superseded by a
duplicate. "Maintainers confirmed and fixed" is already accurate and should **not** be
strengthened to a claim that the authors' patch was merged.
