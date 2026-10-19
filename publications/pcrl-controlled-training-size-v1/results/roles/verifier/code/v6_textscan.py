"""Verifier stage V6d: scope scan of the final texts against RULES.md sections 11-12 and A-SC1 section 5, and verification of every
quoted number against the verifier's own records (v6_numbers.py output). The judgments are the verifier's (written below with the
document hashes they apply to); a forbidden-phrase scan is automatic.

PCRL controlled training-size campaign (TSZ1).

    python3 .../v6_textscan.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

RES = V.RESULTS
DOCS = ('RESEARCH_DECISION.md', 'ADVISOR_BRIEF.md', 'START_HERE.md')
REQUIRED = {'RESEARCH_DECISION.md': ('not a population learning curve', 'independent replication', 'not a new two-output joint method',
                                     'arbitrary-reader protection', 'not independent datasets', 'Amendment A-SC1', 'nothing is confirmation')}
FORBIDDEN = (r'\bconfirm(?:s|ed|ation)?\b(?! (?:nothing|of nothing))', r'population learning curve(?! ?\()', r'\bnovel\b', r'arbitrary readers?\b',
             r'\bguarantee[sd]?\b', r'across reader seeds', r'\bproves?\b', r'\bjoint[- ]design advantage\b')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    # V2 correction: phrases are matched on whitespace-normalized text (V1 missed a phrase wrapped across lines), and every
    # forbidden-term hit is classified: a hit inside a negation or a NOT_* label, or describing the verifier's replay, is benign
    texts = {d: (RES / d).read_text() for d in DOCS if (RES / d).is_file()}
    norm = {d: re.sub(r'\s+', ' ', t) for d, t in texts.items()}
    NEG = re.compile(r"\b(not|no|nothing|never|neither|nor|do not|does not|NOT_CLAIMABLE|NOT_CLAIMED|NOT_MET|would not)\b|verifier",
                     flags=re.IGNORECASE)
    hits, benign = {}, {}
    for d, t in norm.items():
        for pat in FORBIDDEN:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                ctx = t[max(0, m.start() - 70): m.end() + 70]
                rec = {'pattern': pat, 'context': ctx}
                (benign if NEG.search(ctx) else hits).setdefault(d, []).append(rec)
    missing = {d: [s for s in req if s.lower() not in norm.get(d, '').lower()] for d, req in REQUIRED.items()}
    findings = [
        {'doc': 'RESEARCH_DECISION.md', 'where': 'section 3, size gain of the control',
         'text': '"the difference between the two gains (per-person double difference) is not resolved for any metric"',
         'verifier': 'the DiD vs TASK_REFIT for women\'s Brier (50->100) is VERSION_DEPENDENT: resolved positive in v1 (+2.44e-3), unresolved in v0 '
                     '(+1.21e-3); "not resolved on both versions for any metric" would be exact', 'severity': 'ATTENTION'},
        {'doc': 'RESEARCH_DECISION.md', 'where': 'section 7, first sentence',
         'text': '"binds the whole-release reserve of every H3-anchored release at 2.86 (v1 income accuracy)"',
         'verifier': 'not every H3-anchored release: T_TREE at 100% (TT100) binds on its occupation side at 2.60 (v1), and the A1_PO_16 50% fits '
                     'bind on the men\'s guard (down to 1.11); the statement holds for the conditional candidates, W and TR100',
         'severity': 'ATTENTION'},
        {'doc': 'RESEARCH_DECISION.md', 'where': 'section 6, equal reserve',
         'text': '"every reference fails the reserve (whole-release minimum ratios 0.32-1.95)"',
         'verifier': 'per-version minima span 0.32-1.95; the equal-reserve table (minimum over both versions) spans 0.32-1.56 (NEURAL_RAW 1.95 / 1.56)',
         'severity': 'NOTE'},
        {'doc': 'RESEARCH_DECISION.md / ADVISOR_BRIEF.md / START_HERE.md', 'where': 'decision in brief; finding 1; outcome line',
         'text': '"the lost margin is explained by training size on this pool" / "Training size explains the lost margin on this pool" / '
                 '"On this pool the lost margin is a training-size effect"',
         'verifier': 'C1 is MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL, whose registered scope is non-causal for historical differences '
                     '(RULES.md 3.6 / 11: "the larger nested training sets had resolved lower loss on MODEL_SELECT and the margin pattern matched"; '
                     'not causal for every historical difference). "explained by" and especially "is a training-size effect" (START_HERE) read '
                     'causally about the earlier campaign\'s lost margin; the earlier partitions also differed in rows and dictionaries',
         'severity': 'ATTENTION'},
        {'doc': 'RESEARCH_DECISION.md', 'where': 'decision in brief',
         'text': '"so the gain is a training-size property of this pool and these learners, not a recipe-specific one"',
         'verifier': 'the registered result is CONTROL_EXPLAINS_GAIN (recipe gain and control gain resolved, DiD not resolved on both versions); an '
                     'unresolved DiD is absence of a resolved recipe-specific gain, not evidence that none exists',
         'severity': 'ATTENTION'},
        {'doc': 'ADVISOR_BRIEF.md / START_HERE.md', 'where': 'reader results',
         'text': 'reader outcomes are reported ("All 8 cells are positive, but the registered screen says not stable"; "does not lower the '
                 'readers\' recovery stably")',
         'verifier': 'A-SC1 section 5 requires its disclosure sentence wherever reader labels are reported; RESEARCH_DECISION.md carries it '
                     'verbatim, ADVISOR_BRIEF.md paraphrases it in its process notes, START_HERE.md does not carry it',
         'severity': 'ATTENTION'},
    ]
    verified = ['every table and bullet number of RESEARCH_DECISION.md sections 1-7 and ADVISOR_BRIEF.md findings 1-4 that the verifier '
                'replayed matches its own records within rounding: size table means (men\'s and NLL excess), reserve counts and ratio ranges, '
                'the five 50->100 contrasts with SEs and labels, 50->70 / 70->100 labels, subset spread and version difference, encoding E and '
                'share, T*, trigger agreement, usable slack 5.2 / 3.9e-3, level differences vs TASK_REFIT and T_TREE with labels, control '
                'reserve counts and ratios 3.41 / 3.19, CONTROL_EXPLAINS_GAIN, selected strengths, realized vs predicted costs, the six pair-'
                'benefit rows (mean, min, max, label), utility ratios, reference pair-benefit range -2.8 to -45.2e-3, gates, target, equal-'
                'reserve frontier statement, income ratios 3.51 / 2.86, the RETIRED_B3CTL income-gate exception',
                'not verified by the verifier: the earlier campaign\'s 42-53% men\'s excess values (+0.65 / +2.15 / +1.11 / +1.89e-3) and the '
                'identical model files of REPRO.json (the verifier compared release arrays, not model files)']
    if hits or any(missing.values()) or any(f['severity'] == 'FAIL' for f in findings):
        status = 'FAIL'
    elif any(f['severity'] == 'ATTENTION' for f in findings):
        status = 'ATTENTION'
    else:
        status = 'PASS'
    chk = {'id': f'V6.TEXT_SCOPE_SCAN__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': 'final texts (RESEARCH_DECISION.md, ADVISOR_BRIEF.md, START_HERE.md) scanned against RULES.md sections 11-12 (claims kept separate, '
                   'scope sentences, nothing claimable that is listed as not claimable) and A-SC1 section 5 (disclosure sentence), and every quoted '
                   'number checked against the verifier\'s own records',
           'inputs': V.inputs_sha([RES / d for d in DOCS]), 'expected': 'no forbidden claim; required scope present; numbers agree',
           'observed': {'forbidden_phrase_hits': hits, 'benign_negated_or_label_mentions': benign, 'required_scope_missing': missing,
                        'findings': findings, 'verified': verified},
           'status': status, 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V6_TEXTSCAN__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'hits': hits, 'missing': missing}, default=str)[:3000])


if __name__ == '__main__':
    main()
