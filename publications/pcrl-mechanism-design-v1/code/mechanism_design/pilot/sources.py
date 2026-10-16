"""Data sources for the prepared pilot: synthetic ACS-shaped stand-ins (used here) and the real source (refused here).

SyntheticSource draws people from an invented generative model with the ACS benchmark's schema: the six permitted raw
fields (AGEP, WKHP, COW, SCHL, MAR, POBP), SEX, income (binary) and five occupation groups, grouped into households,
split into the development roles by household with the admitted role sizes (optionally scaled down), and encoded with
the benchmark's rule (AGEP and WKHP z-scored with MODEL_FIT moments; one-hot COW/SCHL/MAR/POBP levels seen at least
10 times in MODEL_FIT; everything else to a per-variable OTHER column). No real value is used anywhere: the role sizes
are admitted metadata (POPULATION_AND_SCHEMA.md) and every distribution is made up.

RealSource maps the same interface onto acs_replication.data.load_role and the frozen comparator bank. It refuses to
construct without the committed authorization record (mechanism_design.pilot.authorization), which this campaign never
writes.
"""
from __future__ import annotations

import numpy as np

from mechanism_design.pilot import authorization

ROLE_SIZES = {'MODEL_FIT': 20000, 'CALIBRATION': 2982, 'MODEL_SELECT': 8001, 'ATTACK_FIT': 8001, 'ATTACK_VALIDATION': 3000,
              'DEV_REPORT': 5943}                        # admitted metadata; SEALED_ASSESSMENT has no export and is refused
ROLES = tuple(ROLE_SIZES)
TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
CAT = ('COW', 'SCHL', 'MAR', 'POBP')
MIN_COUNT = 10


class SyntheticSource:
    kind = 'synthetic'

    def __init__(self, scale=0.1, seed=20261016):
        self.rng = np.random.default_rng(seed)
        self.sizes = {r: max(200, int(round(n * scale))) for r, n in ROLE_SIZES.items()}
        self.raw, self.truth = {}, {}
        next_unit = 0
        for role in ROLES:
            n = self.sizes[role]
            raw, truth, units = self._people(n, next_unit)
            next_unit = int(units.max()) + 1
            self.raw[role] = {**raw, 'unit': units, 'row_id': np.arange(n) + 10 ** 6 * (ROLES.index(role) + 1)}
            self.truth[role] = truth
        self._fit_encoder()

    def _people(self, n, unit0):
        r = self.rng
        sex = (r.random(n) < 0.47).astype(np.int64)                       # 1 = female
        agep = r.integers(25, 65, n)
        wkhp = np.clip(np.where(sex == 1, r.normal(37.5, 10, n), r.normal(43.5, 8.5, n)).round(), 1, 99).astype(np.int64)
        cow = np.array([r.choice(8, p=(p := np.array([.62, .07, .06, .05, .09, .06, .03, .02]) if s == 0 else np.array([.60, .09, .10, .07, .04, .05, .03, .02]))) + 1 for s in sex])
        schl = np.clip(np.where(sex == 1, r.normal(19.2, 2.6, n), r.normal(18.7, 2.8, n)).round(), 1, 24).astype(np.int64)
        mar = np.array([r.choice(5, p=np.array([.55, .03, .14, .03, .25]) if s == 0 else np.array([.50, .07, .17, .04, .22])) + 1 for s in sex])
        pob_codes = np.array([12, 13, 36, 34, 48, 6, 17, 72, 303, 110, 210, 207, 233, 301, 312, 330, 339, 364, 411, 427, 501, 999])
        pob_p = np.array([.38, .05, .06, .03, .03, .02, .03, .04, .05, .03, .02, .02, .02, .03, .02, .02, .02, .02, .01, .01, .005, .095])
        pobp = pob_codes[r.choice(len(pob_codes), n, p=pob_p / pob_p.sum())]
        h = (wkhp - 40) / 10.0; e = (schl - 19) / 3.0; a = (agep - 45) / 10.0
        li = -0.4 + 1.1 * h + 0.9 * e + 0.35 * a - 0.55 * sex + 0.25 * (mar == 1) - 0.3 * (cow == 6)
        income = (r.random(n) < 1 / (1 + np.exp(-li))).astype(np.int64)
        z = np.column_stack((0.9 * e + 0.1 * a, 0.3 - 0.5 * h + 0.6 * sex - 0.3 * e, 0.4 + 0.9 * sex - 0.1 * e,
                             -0.2 + 0.4 * h - 1.6 * sex - 0.5 * e, 0.1 + 0.3 * h - 1.0 * sex - 0.6 * e))
        pz = np.exp(z - z.max(1, keepdims=True)); pz /= pz.sum(1, keepdims=True)
        occ = np.array([r.choice(5, p=p) for p in pz], dtype=np.int64)
        sizes = r.choice([1, 2, 3], size=n, p=[.5, .35, .15])
        units = np.repeat(np.arange(unit0, unit0 + n), sizes)[:n]
        raw = {'AGEP': agep, 'WKHP': wkhp, 'COW': cow.astype(str), 'SCHL': np.char.zfill(schl.astype(str), 2), 'MAR': mar.astype(str),
               'POBP': np.char.zfill(pobp.astype(str), 3)}
        return raw, {'sex': sex, 'income': income, 'occupation': occ}, units

    def _fit_encoder(self):
        mf = self.raw['MODEL_FIT']
        self.moments = {c: (float(np.mean(mf[c])), float(np.std(mf[c]))) for c in ('AGEP', 'WKHP')}
        self.levels = {}
        for c in CAT:
            vals, counts = np.unique(mf[c], return_counts=True)
            self.levels[c] = sorted(v for v, k in zip(vals, counts) if k >= MIN_COUNT)
        self.feature_names = ['AGEP', 'WKHP'] + [f'{c}={v}' for c in CAT for v in self.levels[c]] + [f'{c}=OTHER' for c in CAT]

    def encode(self, raw):
        cols = [(raw[c] - self.moments[c][0]) / self.moments[c][1] for c in ('AGEP', 'WKHP')]
        for c in CAT:
            cols += [(raw[c] == v).astype(np.float64) for v in self.levels[c]]
        for c in CAT:
            cols.append((~np.isin(raw[c], self.levels[c])).astype(np.float64))
        return np.column_stack(cols).astype(np.float32)

    def load_role(self, role, wanted=('features',)):
        authorization.refuse_role(role)
        r = self.raw[role]
        out = {'X': self.encode(r), 'feature_names': tuple(self.feature_names), 'unit': r['unit'], 'row_id': r['row_id']}
        for w in wanted:
            if w in ('sex', 'income', 'occupation'):
                if role == 'DEV_REPORT':
                    raise PermissionError('DEV_REPORT truth only through dev_report_truth()')
                out[w] = self.truth[role][w]
        return out

    def dev_report_truth(self):
        """Synthetic stand-in for the logged truth read; the real source requires the committed pilot REPORTING_LOCK."""
        return {k: v.copy() for k, v in self.truth['DEV_REPORT'].items()}


class RealSource:
    kind = 'real'

    def __init__(self, protocol_sha256):
        self.auth = authorization.require_real_data_authorization(protocol_sha256)   # refuses in this campaign
        from acs_replication import data as D                                          # pragma: no cover
        self.D = D

    def load_role(self, role, wanted=('features',)):                                  # pragma: no cover
        authorization.refuse_role(role)
        return self.D.load_role(role, wanted)

    def dev_report_truth(self):                                                       # pragma: no cover
        authorization.require_reporting_lock()
        return self.D.load_role('DEV_REPORT', ('features', 'sex', 'income', 'occupation'))
