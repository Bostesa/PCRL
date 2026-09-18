"""Phase 1: diagnose the completed direct-adversarial run's checkpoint selection.

This stage runs **entirely on stored records**. It refits nothing, re-audits nothing
and opens no 2018 or 2017 outcome. Its inputs are the 114 `fit_record.json` files and
their `checkpoints.pt` parameter states in the read-only adversarial worktree.

What it establishes, and what it cannot:

* It reconstructs `C_t` (source cross-entropy), `D_t` (teacher distortion, already
  normalised by the fixed training variance) and `P_t` (the policy penalty) at every
  checkpoint from the stored **equal-budget fresh-probe** monitor records, and verifies
  that the published selected index is exactly `argmin_t (C_t + D_t + beta P_t)` with
  ties to the earlier step.
* It measures whether the trajectory **moved** before its selected step, in three
  independent currencies: normalised teacher distortion, raw parameter displacement,
  and the released-channel monitor score gap.
* It **counterfactually rescores the same stored trajectories** at
  `gamma in {0, .01, .1, 1}`. This changes **selection only**. It is emphatically NOT
  a utility ablation: it cannot say what retraining at a different `gamma` would do,
  because the trajectory it selects from was itself produced under `gamma = 1`.
* It separates three candidate explanations for a frequently-active zero-gain option
  (`G_j = max(0, max_k g_jk) = 0`) as far as the stored evidence allows, and keeps the
  residual uncertainty where the evidence cannot separate them.

No residence and no commute label is read anywhere in this module.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import (ADVERSARIAL_RESULTS, LOGS, OUT, Stopwatch, limit_threads, read_json,
                     sha_file, utcnow, write_json_atomic)

GAMMA_GRID = (0.0, 0.01, 0.1, 1.0)
SEEDS = (0, 1, 2)

# The fresh-probe record is the registered selection yardstick (RUN_STATUS amendment 1).
# The other two stored slates are kept for the budget-mismatch analysis and are never
# used to select.
SLATES = ('monitor_scores', 'final_slate_monitor_scores', 'contemporaneous_monitor_scores')


def fit_records(root: Path = ADVERSARIAL_RESULTS, seeds=SEEDS):
    for seed in seeds:
        directory = root / f'seed_{seed}' / 'fits'
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            record = path / 'fit_record.json'
            if record.exists():
                yield seed, path.name, read_json(record), path


def components(entry: dict) -> tuple:
    """`(C_t, D_t, P_t)` from one stored monitor record.

    `utility = source + teacher_weight * distortion` in the implementation, and the
    stored `source` and `distortion` are the two terms separately, so no arithmetic
    inversion is needed and the identity is checked rather than assumed.
    """
    return float(entry['source']), float(entry['distortion']), float(entry['penalty'])


def verify_identity(record: dict, tolerance: float = 5e-6) -> dict:
    """`utility == source + w*distortion` and `monitor_score == utility + beta*penalty`.

    A failure here would mean the stored components do not reconstruct the score the
    run actually selected on, and every downstream counterfactual would be invalid.
    """
    beta = float(record['beta'])
    weight = float(record['config']['teacher_weight'])
    worst_utility, worst_score = 0.0, 0.0
    for slate in SLATES:
        for entry in record[slate]:
            source, distortion, penalty = components(entry)
            worst_utility = max(worst_utility,
                                abs(entry['utility'] - (source + weight * distortion)))
            worst_score = max(worst_score,
                              abs(entry['monitor_score'] - (entry['utility'] + beta * penalty)))
    return {'max_abs_utility_residual': worst_utility,
            'max_abs_score_residual': worst_score,
            'implementation_weight': weight, 'beta': beta,
            'passes': bool(worst_utility <= tolerance and worst_score <= tolerance)}


def rescore(record: dict, gamma: float) -> dict:
    """`argmin_t (C_t + gamma D_t + beta P_t)` over the SAME stored trajectory."""
    beta = float(record['beta'])
    scores, steps = [], []
    for entry in record['monitor_scores']:
        source, distortion, penalty = components(entry)
        scores.append(source + gamma * distortion + beta * penalty)
        steps.append(int(entry['step']))
    scores = np.asarray(scores, float)
    index = int(np.argmin(scores))                    # numpy argmin: first minimiser == earlier step
    order = np.argsort(scores)
    margin = float(scores[order[1]] - scores[order[0]]) if len(scores) > 1 else float('nan')
    return {'gamma': gamma, 'selected_index': index, 'selected_step': steps[index],
            'scores': [float(s) for s in scores], 'steps': steps,
            'margin_to_runner_up': margin,
            'score_at_step0': float(scores[0]),
            'score_at_selected': float(scores[index]),
            'step0_disadvantage': float(scores[0] - scores[index])}


def parameter_displacement(path: Path) -> dict:
    """L2 displacement of the mapper/head parameters from step 0, per checkpoint.

    Distortion measures movement of the OUTPUT on the monitor fold; this measures
    movement of the parameters. A trajectory that is genuinely stationary is stationary
    in both, and a trajectory that moved but was not selected shows up here even if its
    output distortion is small.
    """
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False)
    states, steps = blob['checkpoint_states'], blob['checkpoint_steps']
    flat = []
    for state in states:
        flat.append(np.concatenate([np.asarray(v, dtype=np.float64).ravel()
                                    for v in state.values()]))
    base = flat[0]
    norm0 = float(np.linalg.norm(base))
    out = {'steps': [int(s) for s in steps],
           'l2_from_step0': [float(np.linalg.norm(f - base)) for f in flat],
           'parameter_norm_step0': norm0}
    out['relative_l2_from_step0'] = [v / norm0 if norm0 > 0 else float('nan')
                                     for v in out['l2_from_step0']]
    out['max_abs_from_step0'] = [float(np.abs(f - base).max()) for f in flat]
    return out


def gain_profile(record: dict) -> dict:
    """How often the `max(0, ·)` zero option is active, and what the raw gains look like.

    Three explanations are separable only in part:

    * the service-only baseline `p0_j` is genuinely strong, so no correction helps;
    * the correction attacker is underfit at the probe's 300-update budget;
    * the achievable gain really is near zero.

    The stored evidence distinguishes the SECOND from the other two, because the same
    checkpoints carry a 3000-update contemporaneous slate and a 300-update fresh slate.
    It does not separate the first from the third, and this function says so by
    reporting both the clamp rate and the unclamped raw margins rather than a verdict.
    """
    profile = {}
    for slate in SLATES:
        clamped, structural, total, raws = 0, 0, 0, []
        per_role_clamped = Counter()
        per_role_structural = Counter()
        per_role_total = Counter()
        for entry in record[slate]:
            active = entry.get('active_attacker', {})
            for role, value in entry['gains'].items():
                total += 1
                per_role_total[role] += 1
                # A role with no attacker slot at all reports a STRUCTURAL zero. That is
                # the policy's definition (L1/L2 spend their slots on local replicas and
                # never fit an AB attacker), not a measurement that recovery is zero.
                if active.get(role) is None:
                    structural += 1
                    per_role_structural[role] += 1
                    continue
                raws.append(float(value))
                if float(value) <= 0.0:
                    clamped += 1
                    per_role_clamped[role] += 1
        measured = total - structural
        profile[slate] = {
            'role_checkpoint_pairs': total,
            'structural_zero_no_attacker_slot': structural,
            'measured_role_checkpoint_pairs': measured,
            'zero_option_active_on_measured': clamped,
            'measured_zero_option_rate': clamped / measured if measured else float('nan'),
            'raw_zero_rate_including_structural': (clamped + structural) / total if total
                                                  else float('nan'),
            'per_role_measured_zero_rate': {
                r: (per_role_clamped[r] / (per_role_total[r] - per_role_structural[r]))
                if per_role_total[r] - per_role_structural[r] else None
                for r in per_role_total},
            'per_role_structural_rate': {r: per_role_structural[r] / per_role_total[r]
                                         for r in per_role_total},
            'gain_mean': float(np.mean(raws)) if raws else float('nan'),
            'gain_max': float(np.max(raws)) if raws else float('nan'),
            'gain_min': float(np.min(raws)) if raws else float('nan'),
            'gain_p10': float(np.percentile(raws, 10)) if raws else float('nan')}
    return profile


def budget_mismatch(record: dict) -> dict:
    """Which stored slate comparisons are invalid because their budgets differ.

    * `monitor_scores` (fresh probe): 300 attacker updates per checkpoint, EQUAL.
    * `contemporaneous_monitor_scores`: the training slate as it stood at that step —
      100 warmup at step 0, rising to warmup + 5*step + catch-ups at the end. UNEQUAL
      by construction, and the inequality grows with the step.
    * `final_slate_monitor_scores`: one slate, trained against the FINAL channel, read
      backwards. Equal in budget but not in target, and biased toward understating what
      an early checkpoint leaks.
    """
    config = record['config']
    warmup = int(config['attacker_warmup'])
    per_mapper = int(config['attacker_updates_per_mapper'])
    catchup = int(config['refresh_catchup'])
    fractions = config['refresh_fractions']
    updates = int(config['mapper_updates'])
    rows = []
    for entry in record['contemporaneous_monitor_scores']:
        step = int(entry['step'])
        done = sum(1 for f in fractions if int(round(float(f) * updates)) <= step)
        rows.append({'step': step,
                     'contemporaneous_attacker_updates': warmup + per_mapper * step
                                                         + catchup * done,
                     'fresh_probe_attacker_updates': int(config['selection_probe_updates'])})
    spread = [r['contemporaneous_attacker_updates'] for r in rows]
    return {'per_checkpoint': rows,
            'contemporaneous_budget_min': min(spread), 'contemporaneous_budget_max': max(spread),
            'contemporaneous_budget_ratio': max(spread) / max(min(spread), 1),
            'fresh_probe_budget': int(config['selection_probe_updates']),
            'valid_for_cross_checkpoint_comparison': ['monitor_scores'],
            'invalid_for_cross_checkpoint_comparison': [
                'contemporaneous_monitor_scores (budget rises with the step by construction; '
                'the measured penalty therefore rises with training even when the channel '
                'does not move)',
                'final_slate_monitor_scores (equal budget but a single slate specialised to '
                'the final channel, which understates early-checkpoint recoverability)']}


def diagnose_one(seed: int, arm: str, record: dict, path: Path, with_parameters: bool) -> dict:
    identity = verify_identity(record)
    published = int(record['selected_index'])
    counterfactual = {f'{g:g}': rescore(record, g) for g in GAMMA_GRID}
    at_one = counterfactual['1']
    entries = record['monitor_scores']
    selected = entries[published]
    initial = entries[0]
    final = entries[-1]

    decomposition = {
        'selected_minus_initial': {
            'source': float(selected['source'] - initial['source']),
            'distortion': float(selected['distortion'] - initial['distortion']),
            'penalty': float(selected['penalty'] - initial['penalty']),
            'beta_times_penalty': float(record['beta'] * (selected['penalty']
                                                          - initial['penalty'])),
            'monitor_score': float(selected['monitor_score'] - initial['monitor_score'])},
        'final_minus_selected': {
            'source': float(final['source'] - selected['source']),
            'distortion': float(final['distortion'] - selected['distortion']),
            'penalty': float(final['penalty'] - selected['penalty']),
            'monitor_score': float(final['monitor_score'] - selected['monitor_score'])}}

    out = {
        'seed': seed, 'arm': arm, 'width': record['width'], 'policy': record['policy'],
        'beta': record['beta'], 'repeat': record.get('repeat', 0),
        'families': record['families'],
        'teacher_weight': record['config']['teacher_weight'],
        'published_selected_index': published,
        'published_selected_step': int(record['selected_step']),
        'reconstruction_matches_published': bool(at_one['selected_index'] == published),
        'component_identity': identity,
        'counterfactual_selection': counterfactual,
        'selection_changes_with_gamma': {
            g: int(counterfactual[g]['selected_step']) for g in counterfactual},
        'decomposition': decomposition,
        'movement': {
            'selected_step_is_zero': bool(record['selected_step'] == 0),
            'distortion_at_selected': float(selected['distortion']),
            'distortion_at_final': float(final['distortion']),
            'max_distortion_on_trajectory': float(max(e['distortion'] for e in entries)),
            'penalty_at_initial': float(initial['penalty']),
            'penalty_at_selected': float(selected['penalty']),
            'penalty_at_final': float(final['penalty']),
            'margin_to_runner_up': at_one['margin_to_runner_up'],
            'step0_disadvantage': at_one['step0_disadvantage']},
        'gain_profile': gain_profile(record),
        'budget': budget_mismatch(record),
        'fit_record_sha256': sha_file(path / 'fit_record.json'),
        'checkpoints_sha256': record['checkpoints_sha256'],
        'runtime_seconds': record['runtime_seconds'],
    }
    if with_parameters and (path / 'checkpoints.pt').exists():
        out['parameter_displacement'] = parameter_displacement(path / 'checkpoints.pt')
        moved = out['parameter_displacement']['relative_l2_from_step0']
        out['movement']['relative_parameter_l2_at_selected'] = moved[published]
        out['movement']['relative_parameter_l2_at_final'] = moved[-1]
        out['movement']['moved_before_selection'] = bool(published > 0 and moved[published] > 0)
        out['movement']['trajectory_moved_at_all'] = bool(max(moved) > 0)
    return out


def near_tie(entry: dict, threshold: float) -> bool:
    return bool(abs(entry['movement']['margin_to_runner_up']) < threshold)


def summarise(units: list, near_tie_threshold: float) -> dict:
    total = len(units)
    step0 = [u for u in units if u['movement']['selected_step_is_zero']]
    moved_but_step0 = [u for u in step0
                       if u['movement'].get('trajectory_moved_at_all', False)]
    changes = defaultdict(Counter)
    switch = defaultdict(int)
    for unit in units:
        base = unit['selection_changes_with_gamma']['1']
        for gamma, step in unit['selection_changes_with_gamma'].items():
            changes[gamma][step] += 1
            if step != base:
                switch[gamma] += 1
    identity_failures = [f"{u['seed']}/{u['arm']}" for u in units
                         if not u['component_identity']['passes']]
    mismatch = [f"{u['seed']}/{u['arm']}" for u in units
                if not u['reconstruction_matches_published']]
    ties = [f"{u['seed']}/{u['arm']}" for u in step0 if near_tie(u, near_tie_threshold)]

    rate = lambda slate, key: float(np.mean([u['gain_profile'][slate][key] for u in units]))
    fresh_rate = rate('monitor_scores', 'measured_zero_option_rate')
    contemporaneous_rate = rate('contemporaneous_monitor_scores', 'measured_zero_option_rate')
    final_rate = rate('final_slate_monitor_scores', 'measured_zero_option_rate')
    structural_rate = rate('monitor_scores', 'raw_zero_rate_including_structural')
    return {
        'units': total,
        'published_selected_step0': len(step0),
        'published_selected_step0_that_did_move': len(moved_but_step0),
        'reconstruction_matches_published': total - len(mismatch),
        'reconstruction_mismatches': mismatch,
        'component_identity_failures': identity_failures,
        'near_tie_threshold': near_tie_threshold,
        'step0_selections_that_were_near_ties': len(ties),
        'step0_near_tie_units': ties,
        'gamma_selection_histogram': {g: dict(sorted(c.items())) for g, c in changes.items()},
        'gamma_units_whose_selection_changes_vs_gamma1': dict(switch),
        'measured_zero_option_rate_by_slate': {
            'fresh_probe_300_updates': fresh_rate,
            'contemporaneous_training_slate_100_to_3220_updates': contemporaneous_rate,
            'final_slate': final_rate},
        'raw_zero_rate_including_structural_fresh_probe': structural_rate,
        'structural_zero_note': ('a role with no attacker slot reports a zero by the policy '
                                 "definition, not by measurement: L1/L2 spend their two "
                                 'coalition-equivalent slots on local replicas and never fit an '
                                 'AB attacker, so every AB entry in a local arm is structural.'),
    }


# ------------------------------------------------------------------ training-trace scales
def trace_analysis(units_source=None, seeds=SEEDS) -> dict:
    """Do the stored gradients and loss scales agree with the declared implementation?

    Three outcome-free checks on the training traces:

    * mapper gradient norms are finite and never identically zero, so the mapper step
      was doing something at every logged step;
    * the penalty measured against the CONTEMPORANEOUS training slate rises at low
      `beta` and is progressively suppressed as `beta` grows — the dose-response the
      declared sign convention predicts;
    * teacher distortion at the final step rises monotonically with `beta`, which is
      what "protection is bought with distortion" means in this objective.

    The contemporaneous slate's budget grows 32x across a trajectory, so a flat penalty
    curve at high `beta` means the channel kept pace with a strengthening attacker, NOT
    that absolute recovery fell. That distinction is the point of the third column.
    """
    grads = []
    grouped = defaultdict(list)
    for seed, arm, record, _path in fit_records(seeds=seeds):
        trace = record['trace']
        grads.extend(float(x['mapper_grad_norm']) for x in trace)
        key = f"{record['policy']}|beta={record['beta']:g}"
        grouped[key].append((float(trace[0]['penalty']), float(trace[-1]['penalty']),
                             float(trace[0]['distortion']), float(trace[-1]['distortion'])))
    array = np.asarray(grads, float)
    dose = {}
    for key, rows in sorted(grouped.items()):
        block = np.asarray(rows, float)
        dose[key] = {
            'arms': int(len(block)),
            'contemporaneous_penalty_first': float(block[:, 0].mean()),
            'contemporaneous_penalty_last': float(block[:, 1].mean()),
            'contemporaneous_penalty_delta': float((block[:, 1] - block[:, 0]).mean()),
            'distortion_first': float(block[:, 2].mean()),
            'distortion_last': float(block[:, 3].mean())}
    return {
        'mapper_grad_norm': {
            'logged_steps': int(array.size), 'all_finite': bool(np.isfinite(array).all()),
            'min': float(array.min()), 'median': float(np.median(array)),
            'max': float(array.max()), 'exact_zeros': int((array == 0).sum())},
        'dose_response': dose,
        'interpretation': (
            'the contemporaneous slate gains 32x its starting budget along a trajectory, so a '
            'penalty that merely stops rising at high beta means the channel kept pace with a '
            'strengthening attacker. It is not evidence that absolute recovery fell, and the '
            '2018 audit is the only thing that speaks to that.')}


def run(out: Path = OUT, seeds=SEEDS, with_parameters: bool = True,
        near_tie_threshold: float = 0.005) -> dict:
    limit_threads()
    watch = Stopwatch('diagnose')
    units = []
    for seed, arm, record, path in fit_records(seeds=seeds):
        units.append(diagnose_one(seed, arm, record, path, with_parameters))
    summary = summarise(units, near_tie_threshold)
    ledger = {
        'generated_utc': utcnow(),
        'source_study': str(ADVERSARIAL_RESULTS),
        'source_commit': '69e790af36c5ca53203dab17b757a8e3415ee934',
        'gamma_grid': list(GAMMA_GRID),
        'scope': ('stored records only: no refit, no re-audit, no 2018 or 2017 outcome opened, '
                  'no residence or commute label read'),
        'counterfactual_note': ('rescoring changes SELECTION ONLY on trajectories that were '
                                'themselves produced under gamma = 1. It is not a utility '
                                'ablation and says nothing about what retraining at another '
                                'gamma would produce.'),
        'summary': summary,
        'trace_analysis': trace_analysis(seeds=seeds),
        'units': units,
        'runtime_seconds': watch.elapsed()}
    write_json_atomic(out / 'CHECKPOINT_LEDGER.json', ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--no-parameters', action='store_true')
    args = parser.parse_args()
    ledger = run(args.out, tuple(args.seeds), not args.no_parameters)
    print(json.dumps(ledger['summary'], indent=1))


if __name__ == '__main__':
    main()
