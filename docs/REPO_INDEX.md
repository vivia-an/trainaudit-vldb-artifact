# sdc_llm_icml_2025 Quick Index

## Project Identity

- Paper title: `TrainAudit: Robust Silent Error Detection for LLM Training`
- Venue/style: ICML 2025 LaTeX project
- Main artifact: `main.tex`
- Build output currently present: `build/main.pdf`
- Main bibliography: `main.bib`

## Core Thesis

The paper argues that silent errors in distributed LLM training are hard to detect because prior systems observe runtime behavior but do not know the intended semantics. `TrainAudit` uses a multi-agent LLM pipeline to read framework source code and documentation, mine topology-aware invariants, validate them adversarially, and enforce them online through SQL over collected runtime states.

Headline numbers used repeatedly in the repo:

- Detection rate: `90.9%`
- Runtime overhead: `~5%`
- TrainCheck detection rate: `6.1%`
- TrainCheck false positive rate on TP workloads: `100%`
- Injected faults: `33`
- Historical bugs: `15`

## High-Value Files

- `main.tex`: full paper source, including appendix at the end
- `PAPER_STORY.md`: Chinese story arc, core contributions, comparison framing
- `CLAUDE.md`: repo-specific editing/build guidance and consistency reminders
- `main.bib`: all references
- `figures/`: paper figures and a few generation scripts
- `appendix.tex`: old backup only, not the active appendix source

## Main.tex Navigation

### Front Matter

- Abstract: `main.tex:226`
- Introduction: `main.tex:236`
- Background and Problem Definition: `main.tex:272`
- System Design: `main.tex:370`
- Evaluation: `main.tex:453`
- Related Work: `main.tex:729`
- Bibliography call: `main.tex:757`
- Appendix start: `main.tex:766`

### Key System Sections

- Invariant Miner: `main.tex:387`
- Data Collector: `main.tex:425`
- Verifier: `main.tex:432`

### Evaluation Subsections

- Detection Efficacy: `main.tex:574`
- Diagnosis Accuracy: `main.tex:659`
- Efficiency and Overhead: `main.tex:679`
- Generalizability and Cross-Framework Portability: `main.tex:703`

### Appendix Sections

- Detailed Constraint Taxonomy: `main.tex:775`
- End-to-End Case Study: `main.tex:790`
- Invariant Miner Algorithm: `main.tex:844`
- LLM Prompt Templates: `main.tex:1015`
- SQL Query Specification: `main.tex:1167`
- Data Collector Details: `main.tex:1252`
- Batch Detection Workflow: `main.tex:1321`
- Extended Experimental Data: `main.tex:1401`

## Important Labels

Use these when editing cross-references:

- Invariant miner reference anchor: `subsec:invariant_miner`
- Verifier reference anchor: `subsec:error-analysis`
- Evaluation section: `sec:evaluation`
- Detection subsection: `subsec:detection`
- Diagnosis subsection: `subsec:diagnosis`
- Efficiency subsection: `subsec:efficiency`
- Scalability subsection: `subsec:scalability`
- Constraint taxonomy appendix: `app:constraint_details`
- Case study appendix: `app:case_study`
- Miner algorithm appendix: `app:algorithm`
- Prompt appendix: `app:prompts`
- SQL appendix: `app:sql`
- Collector appendix: `app:collector`
- Batch appendix: `app:batch`
- Extended data appendix: `app:extended_data`

## Figure Entry Points

Current figure inclusions in the active paper:

- `figures/fig1_clean.pdf`: intro motivating example
- `figures/Diagram-drawio.png`: system overview
- `figures/agents-cue2.png`: invariant miner workflow
- `figures/reproduced_chart.pdf`: main detection evaluation
- `figures/analysis-time-overhead.pdf`: efficiency/overhead
- `figures/scalability_results_crisp_morandi—1.pdf`: scalability/generalizability
- `figures/constraint-tree.png`: appendix taxonomy figure

## Fast Routing For Common Requests

- Modify abstract or contribution framing: start at `main.tex:226` and `PAPER_STORY.md`
- Rewrite intro/motivation/storyline: start at `main.tex:236` and `PAPER_STORY.md`
- Edit system design details: start at `main.tex:370`
- Update metrics/numbers: check `main.tex`, `CLAUDE.md`, and any repeated mentions in appendix
- Adjust evaluation claims: start at `main.tex:453`
- Update related work/citations: start at `main.tex:729` and `main.bib`
- Add or fix appendix details: start at `main.tex:766`
- Replace or inspect figures: start in `figures/` and search `\\includegraphics` in `main.tex`

## Bibliography Notes

- Historical bug keys: `B1` to `B15`
- Competitive/adjacent systems visible in `main.bib` include `traincheck`, `ttrace`, and `trainverify`
- New citations should be manually checked against source metadata; this repo explicitly warns about citation hallucination

## Build Commands

From repo root:

```bash
xelatex -shell-escape main.tex && bibtex main && xelatex -shell-escape main.tex && xelatex -shell-escape main.tex
```

Quick compile:

```bash
xelatex -shell-escape main.tex
```

`-shell-escape` is required because the paper uses `minted`.

## Working Assumptions For Future Edits

- The active appendix is embedded in `main.tex`, not maintained separately in `appendix.tex`
- `icml2025.sty` and `icml2025.bst` should not be modified
- Repo-specific consistency constraints are documented in `CLAUDE.md`
- Story and positioning guidance are documented in `PAPER_STORY.md`
