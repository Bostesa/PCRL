"""Metadata-only construction of the pending verifier patch; never apply it."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = ROOT / 'occupation_bottleneck_independent/replay.py'


def apply_to_text(original, diff):
    source = original.splitlines(True)
    output = []
    index = 0
    in_hunk = False
    for line in diff.splitlines(True):
        if line.startswith(('--- ', '+++ ')):
            continue
        if line.startswith('@@ '):
            start = int(re.match(r'@@ -(\d+)', line).group(1)) - 1
            output.extend(source[index:start]); index = start; in_hunk = True
        elif in_hunk:
            if line.startswith((' ', '-')):
                if source[index] != line[1:]:
                    raise ValueError('original patch base differs')
                if line.startswith(' '):
                    output.append(source[index])
                index += 1
            elif line.startswith('+'):
                output.append(line[1:])
            elif not line.startswith('\\'):
                raise ValueError('unsupported patch line')
    output.extend(source[index:])
    return ''.join(output)


def main():
    original = TARGET.read_text()
    base_pins = json.loads((HERE / 'REPLAY_PATCH_PINS.json').read_text())
    if hashlib.sha256(original.encode()).hexdigest() != base_pins['base_code_sha256']:
        raise ValueError('live replay differs from pending patch base')
    proposed = apply_to_text(original, (HERE / 'REPLAY_PATCH.diff').read_text())
    functions = (HERE / 'HEADLINE_FUNCTIONS.txt').read_text()
    if 'NotImplementedError' in functions:
        raise ValueError('final reporting adapter not yet supplied')
    proposed = proposed.replace('\ndef assessment(inputs):', '\n' + functions + '\ndef assessment(inputs):')
    proposed = proposed.replace("'audit':audit,'assessment':assessment", "'audit':audit,'assessment':assessment,'headlines':headlines")
    old = """        if entry['canonical_model_alias'].startswith('U0_S'):
            actual=torch_load(inputs.path({'path':entry['checkpoint'],'sha256':entry['checkpoint_sha256']}))
            if not all(torch.equal(v,models[entry['seed']]['model'][k]) for k,v in actual['model'].items()):raise AssertionError('epoch-zero source tensor alias differs')
            aliases.append(entry['unit_id'])"""
    new = """        actual=torch_load(inputs.path({'path':entry['checkpoint'],'sha256':entry['checkpoint_sha256']}))
        source=models[entry['seed']]['model']
        if actual.get('schema')!='TCR1-trainable-own-head-checkpoint-v1' or actual.get('seed')!=entry['seed'] or actual.get('class_orders')!=[[0,1],list(range(6))]:raise AssertionError('admitted checkpoint schema/seed/order differs')
        if tuple(actual.get('feature_names',[]))!=tuple(map(str,roles['OSF_DEFENSE_FIT']['feature_names'])) or set(actual['model'])!=set(source):raise AssertionError('admitted checkpoint feature/parameter schema differs')
        if any(v.shape!=source[k].shape or not torch.isfinite(v).all() for k,v in actual['model'].items()):raise AssertionError('admitted checkpoint parameter shape/finiteness differs')
        if entry['canonical_model_alias'].startswith('U0_S'):
            if not all(torch.equal(v,source[k]) for k,v in actual['model'].items()):raise AssertionError('epoch-zero source tensor alias differs')
            aliases.append(entry['unit_id'])"""
    if old not in proposed:
        raise ValueError('source audit expected block differs')
    proposed = proposed.replace(old, new)
    old = """        result['banks'].append({'arm_id':item['arm_id'],'seed':item['seed'],'recovery':actual,'exact_ignore_other_aliases':'PASS'})"""
    new = """        controls=control_replay(scores,y,a['unit'],2026101206 if item.get('ATT_validation',False) else 2026101207,ix)
        if item.get('ATT_validation',False):compare_controls(controls,freeze.get('ATT_validation_controls'),'ATT control replay')
        result['banks'].append({'arm_id':item['arm_id'],'seed':item['seed'],'recovery':actual,'controls':controls,'exact_ignore_other_aliases':'PASS'})"""
    if old not in proposed:
        raise ValueError('bank audit expected block differs')
    proposed = proposed.replace(old, new)
    ast.parse(proposed)
    diff = ''.join(difflib.unified_diff(original.splitlines(True), proposed.splitlines(True),
                   fromfile='a/occupation_bottleneck_independent/replay.py',
                   tofile='b/occupation_bottleneck_independent/replay.py'))
    (HERE / 'FINAL_REPLAY_PATCH.diff').write_text(diff)
    (HERE / 'REPLAY_PROPOSED.txt').write_text(proposed)
    (HERE / 'FINAL_REPLAY_PATCH_PINS.json').write_text(json.dumps({
        'base_code_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'proposed_code_sha256': hashlib.sha256(proposed.encode()).hexdigest(),
        'patch_sha256': hashlib.sha256(diff.encode()).hexdigest(),
        'existing_assessment_arithmetic_changes': False,
        'new_final_headline_replay': True,
        'numerical_verification': 'ROOT queue required before completion',
        'source_untouched': True,
    }, indent=2) + '\n')
    print(json.dumps({'status': 'FINAL_PATCH_PREPARED_SOURCE_UNTOUCHED', 'proposed_code_sha256': hashlib.sha256(proposed.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
