"""Independent frozen contexts, derived from written R3 specification."""
from __future__ import annotations
import math
import numpy as np


def context_cells(income, occupation):
    cells = []
    for p1, p2 in zip(income, occupation, strict=True):
        if len(p1) != 2 or len(p2) != 6:
            raise ValueError("wrong frozen task classes")
        if abs(math.fsum(p1)-1)>1e-12 or abs(math.fsum(p2)-1)>1e-12:
            raise ValueError("invalid frozen probability")
        margin = abs(float(p1[1])-float(p1[0]))
        entropy = -math.fsum(float(p)*math.log(float(p)) for p in p2 if p>0)/math.log(6)
        mb = sum(margin >= boundary for boundary in (.25,.5,.75))
        eb = int(entropy >= .5)
        cl1 = max(range(2), key=lambda i: p1[i])
        cl2 = max(range(6), key=lambda i: p2[i])
        cells.append((6*cl1+cl2)*8+mb*2+eb)
    return np.asarray(cells, dtype=np.int64)


def hierarchy_from_counts(counts, floor=64):
    """Independent counts-only sibling merge and class-pair fallback."""
    counts = list(map(int, counts))
    if len(counts) != 96 or min(counts)<0:
        raise ValueError("invalid fixed-bin counts")
    mapping = [-1]*96; labels=[]; merged=[]
    for pair in range(12):
        index = 8*pair
        parent_counts = [sum(counts[index+2*m:index+2*m+2]) for m in range(4)]
        total = sum(parent_counts)
        if total < floor:
            continue
        if min(parent_counts)<floor:
            mapping[index:index+8] = [len(labels)]*8
            labels.append(f"classpair/{pair}"); merged.append(total)
        else:
            for margin in range(4):
                lo = index+2*margin
                if min(counts[lo:lo+2]) < floor:
                    mapping[lo:lo+2] = [len(labels)]*2
                    labels.append(f"classpair/{pair}/margin/{margin}")
                    merged.append(parent_counts[margin])
                else:
                    for entropy in range(2):
                        mapping[lo+entropy] = len(labels)
                        labels.append(f"classpair/{pair}/margin/{margin}/entropy/{entropy}")
                        merged.append(counts[lo+entropy])
    return {"raw_to_context": mapping, "labels": labels, "merged_counts": merged}


def independent_mixture(codes, donor_rows, context, theta, scalar=False):
    """Context-shared allocation; unsupported cells are identity for G and C."""
    result=[]
    for original, donor, ctx in zip(codes, donor_rows, context, strict=True):
        allocation = 0. if ctx<0 else float(theta[0 if scalar else ctx])
        if allocation < 0 or allocation > 1:
            raise ValueError("allocation outside [0,1]")
        emitted={tuple(original): 1-allocation}
        for wire, mass in donor.items():
            emitted[tuple(wire)] = emitted.get(tuple(wire),0.)+allocation*mass
        result.append(emitted)
    return result
