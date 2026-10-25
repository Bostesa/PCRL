"""CHECK6 (static part): the verifier's line-by-line review of where labels are read in edg/estimators.py, edg/simulate.py and
edg/bank.py. Every cited line is verified programmatically to contain the quoted text (so a citation cannot drift), and the
reviewed files' SHA-256 are recorded and compared with the registration code lock (GRID.json)."""
import json

import vlib as V

T0 = V.utc_now()
CODE = V.D / 'code' / 'edg'
grid_lock = json.load(open(V.D / 'GRID.json'))['code_sha256']
R = [  # (file, line, quoted text, finding)
    ('estimators.py', 77, 'val = float(SV.auc_star(m1, m0))',
     'C_same: drf.solver.auc_star on the SAME sample\'s per-wire class counts - same-sample by design (the estimator under test).'),
    ('estimators.py', 91, 'v, inf = c_same(ids[m], y[m], K)',
     'C_nested: each fold\'s OWN empirical optimum on that fold only (DRF1 heldout_exact_auc_mean) - same-sample within fold by design.'),
    ('estimators.py', 96, 'return float(np.average([v for _, v in vals], weights=[k for k, _ in vals])), IF',
     'C_nested: people-weighted mean of the fold values.'),
    ('estimators.py', 104, 'tr, te = folds != f, folds == f', 'D_nested: training rows exclude the scored fold.'),
    ('estimators.py', 105, 'look[te] = PRC.lookup_scores(ids[tr], y[tr], ids[te], a=NESTED_A)',
     'D_nested: LOOKUP table from training-fold labels only; the held-out fold is only scored.'),
    ('estimators.py', 106, 's_all = PRC.lookup_scores(ids[tr], y[tr], np.arange(K), a=NESTED_A)',
     'D_nested: the same training-fold table on every wire, for its known-law population AUC (no label).'),
    ('estimators.py', 112, 'pooled = PRC.auc(look, y)', 'D_nested: labels enter once, to score the pooled held-out ranking.'),
    ('estimators.py', 127, 'fit, val, sc = parts <= 4, (parts == 5) | (parts == 6), parts >= 7', 'D_bank: disjoint FIT / VALIDATION / SCORE.'),
    ('estimators.py', 128, 'yf = y[fit]', 'D_bank: table labels are FIT labels only.'),
    ('estimators.py', 134, 's_w = np.where(nw > 0, (n1w + BANK_A * pi_hat) / (nw + BANK_A), pi_hat)',
     'D_bank: pseudo-count 2 toward the FIT rate; unseen key -> FIT rate.'),
    ('estimators.py', 139, 'rec = fit_lookup([keys[i] for i in np.flatnonzero(fit)], yf)',
     'D_bank source check: the source recipe is fitted on FIT keys / labels only.'),
    ('estimators.py', 147, 'sign = orientation(yv, raw[val])', 'D_bank: orientation from VALIDATION labels only.'),
    ('estimators.py', 152, 'v_src = INF_AUC(ys, s[sc])', 'D_bank: SCORE labels used once, to score the oriented table.'),
    ('estimators.py', 156, 'pop = FL.auc_of_score(orient(s_w, sign), m1p, m0p)', 'D_bank: population AUC from the known law (no label).'),
    ('estimators.py', 162, 'a, IF = auc_if(post[ids], y)', 'B: the known posterior of the merged wire; labels only score it.'),
    ('simulate.py', 60, "c, s, unit = L.draw(law, sc['n'], rng)", 'study sample'),
    ('simulate.py', 61, "ct, st, unit_t = L.draw(law, sc['n'], rng)", 'independent test sample (B_indep only), a separate draw'),
    ('simulate.py', 63, 'folds = (rng.permutation(G) % E.N_FOLDS)[unit]', 'folds by unit (household in the clustered cell)'),
    ('simulate.py', 64, 'parts = (rng.permutation(G) % 10)[unit]', 'FIT / VALIDATION / SCORE parts by unit, independent permutation'),
    ('simulate.py', 98, "unit_of = {'C_same': 'study', 'C_nested': 'study', 'D_nested': 'study', 'B_same': 'study', 'D_bank': 'score', 'B_indep': 'test'}",
     'paired SE units: D_bank over SCORE units, B_indep over test units'),
    ('bank.py', 127, 'pad = np.zeros(K, dtype=bool)', 'support rows: never FIT, never VALIDATION'),
    ('bank.py', 128, 'sex = np.concatenate((s, np.zeros(K, dtype=s.dtype)))', 'support rows carry a dummy label that no mask selects'),
    ('bank.py', 129, 'fitm, valm = np.concatenate((fit, pad)), np.concatenate((val, pad))', 'masks padded with False'),
    ('bank.py', 137, 'rd[name] = (sc_all[:n], info)', 'sample-row scores'),
    ('bank.py', 138, "pops[name] = float(FL.auc_of_score(sc_all[n:], V['m1'], V['m0']))", 'support-row scores -> exact population AUC'),
    ('bank.py', 73, 'm = fit_lookup([keys[i] for i in np.flatnonzero(fit)], sex[fit])', 'LOOKUP fitted on FIT rows only'),
    ('bank.py', 80, 'm = fit_reader(X[fit], sex[fit], name, rs)', 'source slate fitted on FIT rows (its scaler too)'),
    ('bank.py', 84, 'tf, _, _ = design_for(name, view, inc[fit], occ[fit])', 'new-recipe feature design fitted on FIT outputs only'),
    ('bank.py', 86, 'm, conv = fit_new(name, X[fit], sex[fit], rs)', 'new recipe (and its scaler) fitted on FIT rows only'),
    ('bank.py', 88, 'sign = orientation(sex[val], raw[val])', 'orientation on VALIDATION only'),
    ('bank.py', 90, "return s, {'val_auc': float(auc(sex[val], s[val])), 'orientation': int(sign), **conv}", 'VALIDATION AUC only'),
    ('bank.py', 96, "return pick([(n, rd[n][1]['val_auc'], f'{view}__{n}', 'source' if n in SRC8 else 'new') for n in RECIPES])",
     'single-view winner: VALIDATION AUC only (ffv.readers.pick: max, exact ties to the earlier candidate)'),
    ('bank.py', 100, "cands += [(n, rd[n][1]['val_auc'], f'pair__{n}', org) for n in group]", 'pair candidates: VALIDATION AUC only'),
    ('bank.py', 180, 'fit, val, scm = parts <= 4, (parts == 5) | (parts == 6), parts >= 7', 'bank uses the grid\'s people and split'),
    ('bank.py', 203, "'winner_score_auc': float(auc(s[scm], s_win[scm])), 'winner_pop_auc': allpop[w['candidate']],",
     'SCORE labels used once, for the selected winner only'),
    ('bank.py', 204, "'best_pop_auc_in_bank': max(allpop.values()), 'best_pop_candidate': max(allpop, key=allpop.get),",
     'oracle (known-law) values: reported in the decomposition, never used to select'),
    ('bank.py', 227, 'rng = np.random.default_rng(np.random.SeedSequence(SIM.BASE_SEED + 2, spawn_key=(sc[\'cell\'], 9999)))',
     'class limits: separate seeds, separate people (no SCORE part)'),
]
cites, bad = [], []
for f, ln, txt, note in R:
    lines = open(CODE / f).read().split('\n')
    ok = txt in lines[ln - 1]
    cites.append({'file': f'code/edg/{f}', 'line': ln, 'quoted': txt, 'verified_present': ok, 'finding': note})
    if not ok:
        bad.append(f'{f}:{ln}')
lock = {f: {'sha256': V.sha256_file(CODE / f), 'equals_registration_lock': V.sha256_file(CODE / f) == grid_lock[f'code/edg/{f}']}
        for f in ('estimators.py', 'simulate.py', 'bank.py', 'laws.py', 'analyze.py', 'source_prc.py', '__init__.py', 'grid.py',
                  'selftest.py')}
conclusions = [
    'No test label chooses a fit, a smoothing value, an orientation or a winner: LOOKUP tables use training-fold (D_nested) or '
    'FIT (D_bank, bank LOOKUP) labels only; every bank recipe and its feature design / scaler is fitted on FIT rows only; '
    'orientation and winner selection use VALIDATION only; SCORE labels are read once, to score the oriented D_bank table and the '
    'selected bank winner; the B_indep test sample is a separate draw used only by B_indep.',
    'The appended support rows (one per observable wire) carry a dummy label and False FIT / VALIDATION masks; they only receive '
    'scores, which give each fitted recipe\'s exact population AUC (confirmed behaviourally in CHECK6b: bitwise-identical fits '
    'with and without them).',
    'C_same and C_nested evaluate the empirical optimum on the sample that estimated it (same-sample by design): this is the '
    'estimator being diagnosed, not a leak.',
    'Behavioural confirmation: CHECK6a_estimators_r2.json (call-argument recording of every lookup_scores call, SCORE- and fold-'
    'label perturbation, exact re-implementations) and CHECK6b_bank.json (SCORE-label perturbation, support-row non-influence, '
    'independent winner rule, population AUCs).',
    'NOTE (not a leak): the D_nested reader is the DRF1 float code; where two cells\' shrunk rates are equal as rationals but '
    'round to different floats it orders them (DIAG6a: 10 such ties in 4 of 52 own-sample views, <= 2.2e-5 AUC). This is the '
    'source reader\'s behaviour, reproduced faithfully.',
    'NOTE: PROTOCOL.md section 5 lists six scenario cells; edg/bank.py BANK_CELLS has five entries because REF|0.5|skewed carries '
    'two bank views (L16, L256) of the same people - consistent with the protocol\'s pairing of cells 2 and 3.',
]
rec = {'check': 6, 'name': 'static_code_review', 'citations': cites, 'citation_failures': bad, 'reviewed_files_vs_registration_lock': lock,
       'conclusions': conclusions,
       'status': 'PASS' if not bad and all(v['equals_registration_lock'] for v in lock.values()) else 'FAIL'}
print(rec['status'], bad, {k: v['equals_registration_lock'] for k, v in lock.items()})
V.write_out('CHECK6_code_review.json', rec, started_utc=T0)
