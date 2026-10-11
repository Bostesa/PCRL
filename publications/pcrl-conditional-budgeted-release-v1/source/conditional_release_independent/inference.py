"""Independent sparse weighted-rank replay for supplied frozen readers."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .arithmetic import validate_group_pairing


def prepare_group_score_mass(score_laws, sex, groups, group_order=None):
    observed = validate_group_pairing(groups, sex)
    order = list(sorted(observed)) if group_order is None else list(group_order)
    if set(order) != set(observed) or len(order)!=len(observed):
        raise ValueError("bootstrap group identity mismatch")
    group_index={g:i for i,g in enumerate(order)}
    scores=sorted(set(s for law in score_laws for s in law))
    score_index={s:i for i,s in enumerate(scores)}
    positive=[]; negative=[]
    for law, label, group in zip(score_laws, sex, groups, strict=True):
        if abs(sum(law.values())-1)>1e-10:
            raise ValueError("score law normalization")
        for score, mass in law.items():
            if not np.isfinite(score) or not np.isfinite(mass) or mass<0:
                raise ValueError("invalid finite score mass")
            (positive if label else negative).append((group_index[group],score_index[score],mass))
    def make(entries):
        if not entries:
            raise ValueError("missing SEX class")
        r,c,v=zip(*entries)
        return sparse.coo_matrix((v,(r,c)),shape=(len(order),len(scores))).tocsr()
    return {"positive":make(positive),"negative":make(negative),"scores":np.asarray(scores),"group_order":order}


def weighted_rank_replicates(prepared, group_counts, batch_size=16):
    counts=np.asarray(group_counts)
    if counts.ndim!=2 or counts.shape[1]!=len(prepared['group_order']) or np.any(counts<0):
        raise ValueError("invalid bootstrap group counts")
    result=[]
    for start in range(0,len(counts),batch_size):
        weights=counts[start:start+batch_size]
        pm=np.asarray(weights@prepared['positive']); nm=np.asarray(weights@prepared['negative'])
        negative_below=np.cumsum(nm,axis=1)-nm
        numerator=np.sum(pm*(negative_below+.5*nm),axis=1)
        denominator=np.sum(pm,axis=1)*np.sum(nm,axis=1)
        if np.any(denominator<=0):
            raise ValueError("bootstrap AUC has absent class")
        result.extend(numerator/denominator)
    return np.asarray(result)


def slate_seed_summary(seed_reader_replicates):
    """Max fixed readers within seed, then equal-seed mean."""
    summaries=[]
    for readers in seed_reader_replicates:
        if not readers:
            raise ValueError("missing reader dependency")
        summaries.append(np.max(np.asarray(readers),axis=0))
    return np.mean(np.asarray(summaries),axis=0)
