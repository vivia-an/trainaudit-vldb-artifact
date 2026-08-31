# Figure 1 Draw.io Optimization Plan

## Goal

Redesign Figure 1 as a cleaner mechanism figure for a paper column, while preserving the same story:

1. The reset path starts with the wrong `micro_step_id`.
2. The shifted ID flips the branch from `Accumulate` to `Copy`.
3. The gradient buffer is overwritten, so `G1` is silently lost.

## Structural Changes

- Split the figure into two balanced panels with one question per panel.
- Panel A shows only the counter-sequence mismatch.
- Panel B shows only the control-flow consequence on the gradient buffer.
- Remove long narrative sentences from the figure body and keep the caption responsible for full explanation.

## Panel Design

### Panel A: Counter Mismatch

- Keep two aligned rows:
  - `Initialization path`
  - `Post-reset path`
- Move the reset event next to the lower row, so the causal relation is local.
- Highlight only the first wrong ID in red.
- Replace the central paragraph-like note with a short two-line comparison:
  - `Expected first ID: 1`
  - `Observed after reset: 0`

### Panel B: Branch Flip and Buffer Overwrite

- Use two symmetric columns:
  - `Micro-step 1`
  - `Micro-step 2`
- Keep the state, decision, and observed action vertically aligned inside each column.
- Show the time advance with a single short arrow labeled `next micro-step`.
- Move the intended behavior into a separate low-emphasis reference box at the bottom.
- Highlight only the erroneous `Copy` branch in red.

## Visual Language

- Blue: normal state/context and non-bug observed flow.
- Red: bug-triggering state and erroneous overwrite result.
- Gray: expected/reference behavior only.
- Amber: control logic elements such as reset and decision diamonds.

## Text Compression Rules

- Keep every node within one or two lines.
- Use nouns and short verb phrases instead of sentence-like annotations.
- Reserve bold text for the bug-critical elements only.

## Output Artifacts

- Editable source: `figures/fig1_optimized.drawio`
- Paper assets: `figures/fig1_optimized.pdf`, `figures/fig1_optimized.svg`, `figures/fig1_optimized.png`
- Regeneration script: `figures/gen_fig1_drawio.py`
