#!/usr/bin/env python3
"""v44 = v43 layout with camera-ready wording and title hierarchy.

All geometry is unchanged. This revision aligns corpus terminology, separates
the SQL witness from the final report, and enlarges only key flow labels.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

_savefig = Figure.savefig
Figure.savefig = lambda *args, **kwargs: None
try:
    import gen_overview_v43 as v43  # noqa: E402
finally:
    Figure.savefig = _savefig


fig, ax = v43.fig, v43.ax

REPLACEMENTS = {
    '35 templates · 13 fault classes': '35 templates · 13 subsystem labels',
    'from 392 silent errors': 'from 392 evidence records',
    '392 confirmed fixes': '392 evidence records',
    'violation report': 'non-empty violation set',
}

KEY_LABELS = {
    'TrainAudit',
    'OFFLINE — mine once per framework',
    'ONLINE — every training job · fully deterministic · no LLM',
    'offline inputs',
    'online output',
    'Invariant Miner',
    'Scope',
    'Ground',
    'Construct',
    'Accept',
    'five-stage FSM',
    'Verified Constraint Library $\\mathcal{P}$',
    'P3 · cross-rank replication',
    'Data Collector',
    'Verifier',
    'at job start',
    'per step, per rule',
    'Violation Report',
    'non-empty violation set',
    'Pattern Catalog',
    'Framework source',
    'acceptance gates',
    'framework adapters',
}

for text in ax.texts:
    label = REPLACEMENTS.get(text.get_text(), text.get_text())
    text.set_text(label)
    if label in KEY_LABELS:
        text.set_fontsize(text.get_fontsize() * 1.10)

fig.savefig(HERE / 'fig_overview_v44.pdf')
fig.savefig(HERE / 'preview' / 'fig_overview_v44.png', dpi=300)
print('done v44: v43 geometry preserved; terminology and hierarchy corrected')
