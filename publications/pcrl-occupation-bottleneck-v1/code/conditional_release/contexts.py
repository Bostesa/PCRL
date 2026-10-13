"""Label-free, frozen-bin conditional allocation contexts.

No labels, identity keys, provider row IDs or fitting-only fallbacks are inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MARGIN_EDGES = (0., .25, .5, .75, 1.)
ENTROPY_EDGES = (0., .5, 1.)
COUNT_FLOOR = 64
RAW_COUNT = 96


def raw_context(probability1, probability2):
    p1, p2 = np.asarray(probability1), np.asarray(probability2)
    if (p1.dtype != np.float64 or p2.dtype != np.float64 or p1.ndim != 2
        or p1.shape[1] != 2 or p2.shape != (len(p1), 6) or not len(p1)
        or not np.isfinite(p1).all() or not np.isfinite(p2).all()
        or np.any(p1 < 0) or np.any(p2 < 0)
        or not np.allclose(p1.sum(1), 1, rtol=0, atol=1e-12)
        or not np.allclose(p2.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("context needs frozen income/occupation probabilities")
    margin = np.abs(p1[:, 1]-p1[:, 0])
    # Zero probability contributes exactly zero entropy.
    positive = p2 > 0
    logs = np.zeros_like(p2); np.log(p2, out=logs, where=positive)
    entropy = -(p2*logs).sum(1)/np.log(6.)
    mb = np.searchsorted(np.asarray(MARGIN_EDGES[1:-1]), margin, side="right")
    eb = np.searchsorted(np.asarray(ENTROPY_EDGES[1:-1]), entropy, side="right")
    pair = 6*p1.argmax(1)+p2.argmax(1)
    return (8*pair+2*mb+eb).astype(np.int64)


@dataclass(frozen=True)
class ContextMap:
    mapping: np.ndarray
    raw_fit_counts: np.ndarray
    merged_fit_counts: np.ndarray
    labels: tuple[str, ...]

    def __post_init__(self):
        m, c, n = map(np.asarray, (self.mapping, self.raw_fit_counts, self.merged_fit_counts))
        if (m.dtype != np.int64 or c.dtype != np.int64 or n.dtype != np.int64
            or m.shape != (RAW_COUNT,) or c.shape != (RAW_COUNT,)
            or n.shape != (len(self.labels),) or np.any(c < 0)
            or np.any(m < -1) or np.any(m >= len(n)) or np.any(n < COUNT_FLOOR)):
            raise ValueError("invalid frozen context map")
        for j in range(len(n)):
            if int(c[m == j].sum()) != int(n[j]):
                raise ValueError("context counts do not bind hierarchy")
        for name, array in (("mapping", m), ("raw_fit_counts", c), ("merged_fit_counts", n)):
            array = array.copy(); array.setflags(write=False); object.__setattr__(self, name, array)

    def apply(self, probability1, probability2):
        return self.mapping[raw_context(probability1, probability2)].copy()

    def report(self):
        return {"schema": "CBR1-label-free-context-map-v1", "initial_cell_count": RAW_COUNT,
                "margin_edges": list(MARGIN_EDGES), "normalized_entropy_edges": list(ENTROPY_EDGES),
                "count_floor_rows": COUNT_FLOOR, "supported_context_count": len(self.labels),
                "supported_fit_rows": int(self.merged_fit_counts.sum()),
                "fallback_fit_rows": int(self.raw_fit_counts[self.mapping < 0].sum()),
                "raw_fit_counts": self.raw_fit_counts.tolist(), "raw_to_context": self.mapping.tolist(),
                "contexts": [{"id": j, "hierarchy": label, "fit_rows": int(self.merged_fit_counts[j])}
                             for j, label in enumerate(self.labels)],
                "merge_order": "entropy siblings, then entire class pair if any margin parent below floor",
                "unsupported_behavior": "exact P0; no class-pair root merge",
                "unseen_supported_bank_pairs": "registered supported context mapping applies"}


def fit_context_map(probability1, probability2):
    raw = raw_context(probability1, probability2)
    counts = np.bincount(raw, minlength=RAW_COUNT).astype(np.int64)
    mapping = np.full(RAW_COUNT, -1, np.int64); labels = []; merged = []
    for pair in range(12):
        start = 8*pair; children = counts[start:start+8].reshape(4, 2)
        parents = children.sum(1)
        if int(parents.sum()) < COUNT_FLOOR:
            continue
        if np.any(parents < COUNT_FLOOR):
            mapping[start:start+8] = len(labels)
            labels.append(f"classpair/{pair}"); merged.append(int(parents.sum()))
            continue
        for margin in range(4):
            i = start+2*margin
            if np.any(children[margin] < COUNT_FLOOR):
                mapping[i:i+2] = len(labels)
                labels.append(f"classpair/{pair}/margin/{margin}")
                merged.append(int(parents[margin]))
            else:
                for entropy in range(2):
                    mapping[i+entropy] = len(labels)
                    labels.append(f"classpair/{pair}/margin/{margin}/entropy/{entropy}")
                    merged.append(int(children[margin, entropy]))
    return ContextMap(mapping, counts, np.asarray(merged, np.int64), tuple(labels))
