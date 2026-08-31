# Artifact Checklist

This file tracks what collaborators need, where it lives, and whether it is safe to put in Git.

## Shared Small Artifacts in Git

| Artifact | Path | Owner | Status | Notes |
|---|---|---|---|---|
| SIGMOD visual plan | `sigmod_revision_plan.html` | Qingsong | INCLUDED | Open directly in a browser. |
| Collaboration entry point | `COLLABORATION_SIGMOD.md` | Qingsong | INCLUDED | Start here. |
| Story register | `story_sigmod.md` | Qingsong | INCLUDED | Source of truth for narrative. |
| Experiment registry | `experiment_registry.md` | Experiment owner | INCLUDED | Source of truth for numbers. |
| Artifact checklist | `artifact_checklist.md` | Junior collaborator | INCLUDED | This file. |
| Current Chinese draft | `main_cn.tex` | Qingsong | INCLUDED | Useful for line references and rewrite. |
| Main English draft | `main.tex` | Qingsong | TRACKED EXISTING | Already in repo. |
| Runbook | `REAL_BUG_DETECTION_RUNBOOK.md` | Experiment owner | INCLUDED | Real bug replay protocol. |
| SIGMOD-related notes | `PAPER_STORY.md`, `REPO_INDEX.md` | Qingsong / junior | INCLUDED | Planning docs. |
| Eval specs | `SPEC_S*.md` | Experiment owner | INCLUDED | S1-S4 task specs. |
| TrainAudit package | `trainaudit/` | Experiment owner | INCLUDED | Small source package. |
| Narrative docs | `docs/` | Qingsong / junior | INCLUDED | Design and evidence notes. |

## Large Artifacts Not in Git

| Artifact | Current Path | Size | Git Status | Sharing Plan |
|---|---|---:|---|---|
| Full benchmark directory | `benchmark/` | ~12G | PARTIAL ONLY | Commit selected top-level scripts/results; keep raw outputs external. |
| Eval directory full tree | `benchmark/eval/` | ~9.9G | PARTIAL ONLY | Commit top-level files only, not deep large outputs. |
| Fault injection full tree | `benchmark/fault_injection/` | ~342M | NOT INCLUDED | Share scripts/results selectively if needed. |
| Framework checkouts | `exp/` | ~9G | NOT INCLUDED | Do not put external repos in this paper repo. |
| Build outputs | `build/`, `tmp/`, `output/` | local | IGNORED | Regenerate locally. |
| Agent/editor state | `.claude/`, `.playwright-cli/`, `.vscode/` | local | IGNORED | Do not share. |

## Related Papers

Do not rely on committing third-party PDFs as the primary sharing mechanism. Use links in notes.

- QURE: https://www.microsoft.com/en-us/research/publication/qure/
- AquaPipe: https://dl.acm.org/doi/10.1145/3709661
- SNAILS SIGMOD PDF: https://adalabucsd.github.io/papers/2025_SNAILS_SIGMOD.pdf
- SNAILS technical report: https://adalabucsd.github.io/papers/TR_2025_SNAILS.pdf

Local downloaded copies may exist under `related_sigmod_papers/`, but collaborators should prefer the links above unless we explicitly decide to version them.

## Consistency Checks to Perform

### Numbers

- [ ] `17/19` Real-SDC detection denominator explained consistently.
- [ ] `0/17` vs `1/17` false positive conflict resolved.
- [ ] `13/14` H200 result either promoted to main result or moved to appendix.
- [ ] `23/23` cross-framework replay relation to Real-SDC explained.
- [ ] `24 active rules`, `45 deployed rules`, and `357 L4-passed candidates` reconciled.

### Text / Formatting

- [ ] Remove or anonymize author names and affiliations for submission.
- [ ] Remove PVLDB block for SIGMOD/PACMMOD submission.
- [ ] Remove placeholders: `XXX`, `URL_TO_YOUR_ARTIFACTS`, `example.com`, funding placeholders.
- [ ] Split appendix into separate supplementary PDF if required.
- [ ] Remove build artifacts and local logs from Git.

### Tables / Figures

- [ ] Add `Training Trace Relations` table.
- [ ] Add `Constraint Catalog Format` table.
- [ ] Add overhead table with real measurements.
- [ ] Add benchmark artifact table for Real-SDC.
- [ ] Confirm every figure source script and data file is recorded.

## Junior Collaborator First Tasks

1. Open `sigmod_revision_plan.html`.
2. Read `story_sigmod.md`.
3. Fill in unchecked items in this file with concrete file/line references.
4. Draft the two new tables in markdown.
5. Create a short memo: "How QURE, AquaPipe, and SNAILS frame AI work as data management."
