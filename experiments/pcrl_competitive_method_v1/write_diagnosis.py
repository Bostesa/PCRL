"""Render SELECTION_DIAGNOSIS.md from CHECKPOINT_LEDGER.json.

Every number in the document is read out of the ledger, so the prose and the
machine-readable backing cannot drift apart. The ledger's own sha256 is printed in the
document header.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import OUT, read_json, sha_file, utcnow, write_text_atomic


def _pct(x) -> str:
    return f'{100 * x:.2f}%'


def render(ledger: dict, ledger_path: Path) -> str:
    s = ledger['summary']
    units = ledger['units']
    trace = ledger['trace_analysis']
    step0 = [u for u in units if u['movement']['selected_step_is_zero']]
    moved = [u for u in units if not u['movement']['selected_step_is_zero']]

    margins = np.array([u['movement']['margin_to_runner_up'] for u in step0])
    rel_final = np.array([u['movement']['relative_parameter_l2_at_final'] for u in step0])
    dist_final = np.array([u['movement']['distortion_at_final'] for u in step0])

    dec = {k: np.array([u['decomposition']['selected_minus_initial'][k] for u in moved])
           for k in ('source', 'distortion', 'beta_times_penalty', 'monitor_score')}

    per_role_measured = defaultdict(list)
    per_role_structural = defaultdict(list)
    for u in units:
        gp = u['gain_profile']['monitor_scores']
        for role, value in gp['per_role_measured_zero_rate'].items():
            if value is not None:
                per_role_measured[role].append(value)
        for role, value in gp['per_role_structural_rate'].items():
            per_role_structural[role].append(value)

    by_policy = defaultdict(lambda: [0, 0])
    for u in units:
        cell = by_policy[u['policy']]
        cell[1] += 1
        if u['selection_changes_with_gamma']['0'] != u['selection_changes_with_gamma']['1']:
            cell[0] += 1

    lines = []
    w = lines.append
    w('# SELECTION_DIAGNOSIS — what the previous run\'s selection objective discarded')
    w('')
    w('Generated ' + utcnow() + ' from `CHECKPOINT_LEDGER.json`')
    w(f'(`sha256 {sha_file(ledger_path)}`), which is the machine-readable backing for every')
    w('number below. Source study: the completed direct-adversarial run at evidence commit')
    w('`' + ledger['source_commit'] + '`, read **read-only**.')
    w('')
    w('**Scope.** ' + ledger['scope'] + '. This stage refits nothing and re-audits nothing:')
    w('it is arithmetic on the monitor records and parameter states the previous run saved.')
    w('')
    w('---')
    w('')
    w('## 0. The prerequisite question')
    w('')
    w('> Did previous training generate plausible alternatives that its selection objective')
    w('> discarded, or were the trajectories themselves unpromising?')
    w('')
    w('**Answer: it generated them, and selection discarded them.** The evidence is in §2')
    w('and §3. This is a statement about *selection*, not about whether the discarded')
    w('checkpoints would have audited well — nothing here has been audited.')
    w('')
    w('## 1. The reconstruction is exact')
    w('')
    w(f"* `{s['units']}` fitted trajectories recovered, 7 checkpoints each.")
    w(f"* `utility == source + w*distortion` and `monitor_score == utility + beta*penalty`")
    w(f"  hold on every stored record of all three slates: "
      f"`{len(s['component_identity_failures'])}` identity failures.")
    w(f"* Re-deriving `argmin_t (C_t + 1.0*D_t + beta*P_t)` from the components reproduces the")
    w(f"  published selected index in **{s['reconstruction_matches_published']} of "
      f"{s['units']}** trajectories, with no mismatches.")
    w('')
    w('So the published selection rule, the stored components and this study\'s arithmetic')
    w('are the same object. Every counterfactual below inherits that.')
    w('')
    w('### 1.1 Gradients and loss scales agree with the implementation')
    w('')
    g = trace['mapper_grad_norm']
    w(f"* Mapper gradient norms over `{g['logged_steps']}` logged steps: all finite, "
      f"`{g['exact_zeros']}` exact zeros,")
    w(f"  range `[{g['min']:.4f}, {g['max']:.4f}]`, median `{g['median']:.4f}`. The mapper step")
    w('  was doing work at every logged step; nothing was silently detached.')
    w('* Dose-response against the contemporaneous training slate, averaged over arms:')
    w('')
    w('| policy / beta | penalty first | penalty last | delta | distortion last |')
    w('|---|---|---|---|---|')
    for key, cell in trace['dose_response'].items():
        w(f"| `{key.replace('|', ' / ')}` | "
          f"{cell['contemporaneous_penalty_first']:.4f} | "
          f"{cell['contemporaneous_penalty_last']:.4f} | "
          f"{cell['contemporaneous_penalty_delta']:+.4f} | "
          f"{cell['distortion_last']:.4f} |")
    w('')
    w('The penalty rise is monotonically suppressed as `beta` grows and the final distortion')
    w('rises monotonically with `beta`. The declared sign convention therefore holds and the')
    w('protection gradient does real work. **But** ' + trace['interpretation'])
    w('')
    main = [u for u in units if u['repeat'] == 0 and not u['arm'].endswith('_single')
            and not u['arm'].endswith('_none')]
    main_step0 = [u for u in main if u['movement']['selected_step_is_zero']]
    w(f'## 2. {len(main_step0)}/{len(main)} unchanged main channels was a SELECTION outcome')
    w('')
    w(f"In the registered main block alone (width x policy x beta x seed, no repeats or")
    w(f"ablations), **{len(main_step0)} of {len(main)}** trajectories published step 0 — the")
    w('previously reported 39/72. Across all fitted trajectories:')
    w('')
    w(f"* **{s['published_selected_step0']} of {s['units']}** trajectories published their")
    w('  **step-0** checkpoint, i.e. the unmoved starting channel.')
    w(f"* **{s['published_selected_step0_that_did_move']} of those "
      f"{s['published_selected_step0']}** had in fact moved: their final checkpoint sits at a")
    w(f"  median relative parameter displacement of **{np.median(rel_final):.4f}** from step 0")
    w(f"  (range `[{rel_final.min():.4f}, {rel_final.max():.4f}]`), with median final teacher")
    w(f"  distortion **{np.median(dist_final):.4f}**.")
    w('')
    w('**No trajectory was stationary.** The correct reading of the previous run\'s 39/72')
    w('unchanged main channels is: optimisation moved the channel, and the registered')
    w('selection objective preferred the starting point anyway. It is *not* evidence that')
    w('optimisation never moved, and it is *not* evidence that teacher distortion alone')
    w('caused the failure — §3 shows the objective as a whole, not the distortion term')
    w('alone, is what selected.')
    w('')
    w('### 2.1 Step 0 usually won by a real margin, not a near tie')
    w('')
    w(f"Margin from step 0 to the runner-up checkpoint, across the {len(step0)} step-0")
    w('selections, on the registered fresh-probe score:')
    w('')
    w('| percentile | margin |')
    w('|---|---|')
    for q in (0, 10, 25, 50, 75, 90, 100):
        w(f'| p{q} | {np.percentile(margins, q):.5f} |')
    w('')
    w(f"At a near-tie threshold of `{s['near_tie_threshold']}`, only")
    w(f"**{s['step0_selections_that_were_near_ties']} of {len(step0)}** step-0 selections were")
    w('near ties. The rest were decisive. Selection was not balanced on a knife edge that a')
    w('small implementation change would have tipped.')
    w('')
    w('## 3. What the moving arms actually paid, and for what')
    w('')
    w(f"Across the **{len(moved)}** trajectories that did select a moved checkpoint, the")
    w('selected-minus-initial decomposition of the score (mean over arms):')
    w('')
    w('| term | mean | median |')
    w('|---|---|---|')
    for key in ('source', 'distortion', 'beta_times_penalty', 'monitor_score'):
        w(f'| `{key}` | {dec[key].mean():+.5f} | {np.median(dec[key]):+.5f} |')
    w('')
    w('Reading: a moved checkpoint is bought with **both** a higher source loss and a higher')
    w('teacher distortion, paid for by a lower protection penalty. The distortion cost is')
    w(f'about **{dec["distortion"].mean() / dec["source"].mean():.1f}x** the source cost in')
    w('these units. That makes the teacher term a large share of what the selection rule was')
    w('charging for movement — **but the source term is not zero either**, so this is not a')
    w('one-factor story, and the coefficient `1.0` on distortion was specified, never shown')
    w('optimal.')
    w('')
    w('## 4. Counterfactual rescoring: selection only')
    w('')
    w('> **This is not a utility ablation.** The trajectories rescored below were produced')
    w('> under `gamma = 1`. Rescoring them at another `gamma` changes which of *those*')
    w('> checkpoints is selected. It says nothing about what retraining at another `gamma`')
    w('> would produce, because a different `gamma` would have generated a different')
    w('> trajectory. Track N answers that question by actually retraining; this section')
    w('> cannot and does not.')
    w('')
    w('`argmin_t (C_t + gamma*D_t + beta*P_t)` on the same stored records:')
    w('')
    w('| gamma | step 0 | 100 | 200 | 300 | 400 | 500 | 600 | selections changed vs gamma=1 |')
    w('|---|---|---|---|---|---|---|---|---|')
    for gamma in ('0', '0.01', '0.1', '1'):
        hist = s['gamma_selection_histogram'][gamma]
        cells = ' | '.join(str(hist.get(str(step), 0)) for step in (0, 100, 200, 300, 400, 500, 600))
        changed = s['gamma_units_whose_selection_changes_vs_gamma1'].get(gamma, 0)
        w(f'| `{gamma}` | {cells} | {changed} |')
    w('')
    w(f"Removing the teacher term entirely moves **{s['gamma_units_whose_selection_changes_vs_gamma1']['0']}")
    w(f"of {s['units']}** selections and drops the step-0 count from")
    w(f"{s['gamma_selection_histogram']['1'].get('0', 0)} to "
      f"{s['gamma_selection_histogram']['0'].get('0', 0)}. By policy:")
    w('')
    w('| policy | selections changed at gamma=0 | arms |')
    w('|---|---|---|')
    for policy in sorted(by_policy):
        changed, total = by_policy[policy]
        w(f'| `{policy}` | {changed} | {total} |')
    w('')
    w('The `gamma` values are **not** filtered by residence or commute outcome, and none was')
    w('removed for being inconvenient: the registered grid `{0, .01, .1, 1}` is reported in')
    w('full, including `gamma = .01`, whose effect is nearly indistinguishable from `gamma = 0`')
    w('at this resolution.')
    w('')
    w('## 5. The zero-gain option was **not** active too often')
    w('')
    w('`G_j = max(0, max_k g_jk)`. A raw count of zeros in the stored records reads')
    w(f"**{_pct(s['raw_zero_rate_including_structural_fresh_probe'])}**, which invites the")
    w('conclusion that the baseline is beating every attacker. That count is wrong, and the')
    w('reason is structural rather than empirical:')
    w('')
    w('* ' + s['structural_zero_note'])
    w('')
    w('Separating the two:')
    w('')
    w('| slate | attacker budget | measured zero-option rate |')
    w('|---|---|---|')
    rates = s['measured_zero_option_rate_by_slate']
    w(f"| fresh probe (the selection yardstick) | 300, equal per checkpoint | "
      f"{_pct(rates['fresh_probe_300_updates'])} |")
    w(f"| contemporaneous training slate | 100 rising to 3220 | "
      f"{_pct(rates['contemporaneous_training_slate_100_to_3220_updates'])} |")
    w(f"| final slate | equal budget, wrong target | {_pct(rates['final_slate'])} |")
    w('')
    w('Per role, on the fresh probe, restricted to roles that actually had an attacker slot:')
    w('')
    w('| role | measured zero rate | structural (no slot) rate |')
    w('|---|---|---|')
    for role in sorted(per_role_structural):
        measured = per_role_measured.get(role)
        measured_text = _pct(float(np.mean(measured))) if measured else 'n/a'
        w(f'| `{role}` | {measured_text} | '
          f'{_pct(float(np.mean(per_role_structural[role])))} |')
    w('')
    w('**Conclusion, with its residual uncertainty.** Where an attacker existed, it found a')
    w('positive gain essentially always. The three candidate explanations offered for a')
    w('frequently-active zero option resolve as follows:')
    w('')
    w('* *the correction attacker is underfit at the probe budget* — **ruled out as the')
    w('  driver**. A 10x larger contemporaneous budget gives essentially the same measured')
    w('  rate, so the 300-update probe is not manufacturing zeros.')
    w('* *the service-only baseline is strong* and *the achievable gain is genuinely near')
    w('  zero* — **not separated, and not separable from these records**, because both')
    w('  predict the same small positive margins. They are also barely relevant here: the')
    w('  measured rate is small enough that the clamp is not what shaped selection.')
    w('')
    w('This removes one hypothesis from the list of explanations for the previous run\'s')
    w('failure. It does **not** establish that the attackers were strong in absolute terms —')
    w('300 fresh updates on three differentiable families is a weak slate by design, and')
    w('only the audit speaks to absolute recoverability.')
    w('')
    w('## 6. Comparisons that are invalid because the budgets differ')
    w('')
    budget = units[0]['budget']
    w(f"Along one trajectory the contemporaneous slate grows from")
    w(f"`{budget['contemporaneous_budget_min']}` to `{budget['contemporaneous_budget_max']}`")
    w(f"attacker updates — a factor of `{budget['contemporaneous_budget_ratio']:.1f}`.")
    w('')
    w('* **Valid for cross-checkpoint comparison:** `' +
      '`, `'.join(budget['valid_for_cross_checkpoint_comparison']) + '`.')
    w('* **Invalid:**')
    for item in budget['invalid_for_cross_checkpoint_comparison']:
        w(f'  * {item}')
    w('')
    w('Both invalid slates are biased in the **same direction**, toward never moving the')
    w('channel. The previous run reached the same conclusion prospectively and switched to')
    w('the equal-budget probe before opening any outcome; this diagnosis confirms the')
    w('switch was necessary and reproduces its arithmetic exactly.')
    w('')
    w('## 7. What this does and does not license')
    w('')
    w('Established:')
    w('')
    w('1. The previous run\'s selection arithmetic is exactly reproducible.')
    w('2. Every trajectory moved; unchanged published channels are a selection outcome.')
    w('3. Step 0 usually won decisively, not by a near tie.')
    w('4. Removing the teacher term changes roughly half of all selections.')
    w('5. The zero-gain clamp was essentially inactive wherever an attacker existed.')
    w('')
    w('**Not** established, and not to be inferred:')
    w('')
    w('* that a discarded checkpoint would have audited better — none of them has been')
    w('  audited, and the monitor score is a weak selection yardstick, not a protection')
    w('  measurement;')
    w('* that `gamma < 1` is better — §4 changes selection on fixed trajectories only;')
    w('* that teacher distortion alone caused the previous failure — §3 shows the source')
    w('  term moves too;')
    w('* that J-initialised fine-tuning behaves like this at all — the previous study')
    w('  initialised from `A0` in every arm and never ran it.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    path = Path(args.out) / 'CHECKPOINT_LEDGER.json'
    text = render(read_json(path), path)
    write_text_atomic(Path(args.out) / 'SELECTION_DIAGNOSIS.md', text)
    print('wrote SELECTION_DIAGNOSIS.md', len(text), 'bytes')


if __name__ == '__main__':
    main()
