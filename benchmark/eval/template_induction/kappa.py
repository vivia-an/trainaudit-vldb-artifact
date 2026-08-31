"""Cohen's kappa and raw agreement helpers (no external deps)."""
from collections import Counter


def cohen_kappa(pairs):
    """pairs: list of (label_a, label_b). Returns (raw_agreement, kappa)."""
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    labels = set(ca) | set(cb)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1.0:
        return po, float("nan")
    return po, (po - pe) / (1 - pe)
