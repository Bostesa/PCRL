"""Diagnostic for the P7 probe: per draw, the field-mediated SEX shift of each within-SEX task logit (b'd, B1'd, B2'd)
versus the direct SEX effect on that logit (-0.45, +0.8, -1.1). Opposite signs mean the inferred-SEX term Delta in the
twin partly cancels the field-mediated SEX signal (masking), so deleting Delta (MMP) can unmask it (verifier M10)."""
from mechanism_design import no_real_data
no_real_data.install()
import json
from pathlib import Path
import numpy as np
from mechanism_design_independent.v_proto_hd import make
OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_PROTO_HD_ALIGN_V1.json'
direct = np.array([-0.45, 0.8, -1.1])
res = {}
for draw in range(3):
    d, b, B = make(draw)
    field = np.array([b @ d, B[0] @ d, B[1] @ d])
    res[f'draw{draw}'] = {'field_mediated_shift': field.tolist(), 'direct_effect': direct.tolist(),
                          'aligned_sign': (np.sign(field) == np.sign(direct)).tolist(),
                          'sum_field_times_direct': float(field @ direct)}
    print(draw, res[f'draw{draw}'])
OUT.write_text(json.dumps(res, indent=1))
