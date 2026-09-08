# Composed teacher attacks

All retained witnesses are frozen teacher-only K-C/K-D models or three-probability source banks followed by their existing auditors. Full-input F models are ineligible. The separately frozen choice uses attacker-validation loss only; person weights score those same choices.

Retained 456 candidate witnesses in 24 teacher/seed/attribute/budget groups. The standalone saved-observer row is diagnostic; epoch zero remains eligible inside catch-up. Full candidate scores and exposure are in [COMPOSED_ATTACKS.csv](COMPOSED_ATTACKS.csv) and [COMPOSED_ATTACKS.json](COMPOSED_ATTACKS.json).

Each mapper has 4,176 stored parameters, including 2,048 inactive raw-input weights; its effective teacher-only mapper has 2,128 parameters and one hidden ReLU layer. Banks add 51 source-head parameters and three sigmoid outputs. The unused 3,168-parameter decoder never enters a predictive composition. JSON records the actual auditor coefficient counts or tree/leaves/depth and all preprocessing stages per witness.

For fixed public parameters on a new example, a(g(T)) equals the stored-release auditor. This extends the teacher witness family with mapper depth and source-label training; D and inherited observers add distinct real-protected-label exposure. It does not replace the teacher's standard direct audit, establish matched budgets, or create information absent from the teacher.

| Teacher | Seed | Attribute | Budget | Scope | Witness | Development loss | PWGTP loss |
|---|---:|---|---:|---|---|---:|---:|
| E | 0 | RAC1P | 120 | independent | B/mlp_1 | 1.259909 | 1.229363 |
| E | 0 | RAC1P | 120 | catchup | D/catchup | 1.208922 | 1.195921 |
| E | 0 | RAC1P | 120 | pooled | D/catchup | 1.208922 | 1.195921 |
| E | 0 | RAC1P | 360 | independent | B/mlp_1 | 1.259909 | 1.229363 |
| E | 0 | RAC1P | 360 | catchup | D/catchup | 1.208922 | 1.195921 |
| E | 0 | RAC1P | 360 | pooled | D/catchup | 1.208922 | 1.195921 |
| E | 0 | SEX | 120 | independent | C/mlp_0 | 0.687210 | 0.691415 |
| E | 0 | SEX | 120 | catchup | D/catchup | 0.676552 | 0.680957 |
| E | 0 | SEX | 120 | pooled | D/catchup | 0.676552 | 0.680957 |
| E | 0 | SEX | 360 | independent | C/mlp_0 | 0.687210 | 0.691415 |
| E | 0 | SEX | 360 | catchup | D/catchup | 0.676552 | 0.680957 |
| E | 0 | SEX | 360 | pooled | D/catchup | 0.676552 | 0.680957 |
| E | 1 | RAC1P | 120 | independent | C/mlp_1 | 1.268235 | 1.292814 |
| E | 1 | RAC1P | 120 | catchup | D/catchup | 1.246328 | 1.274696 |
| E | 1 | RAC1P | 120 | pooled | D/catchup | 1.246328 | 1.274696 |
| E | 1 | RAC1P | 360 | independent | C/mlp_1 | 1.268235 | 1.292814 |
| E | 1 | RAC1P | 360 | catchup | D/catchup | 1.246328 | 1.274696 |
| E | 1 | RAC1P | 360 | pooled | D/catchup | 1.246328 | 1.274696 |
| E | 1 | SEX | 120 | independent | C/mlp_1 | 0.680245 | 0.682650 |
| E | 1 | SEX | 120 | catchup | C/catchup | 0.665039 | 0.668980 |
| E | 1 | SEX | 120 | pooled | C/catchup | 0.665039 | 0.668980 |
| E | 1 | SEX | 360 | independent | C/mlp_1 | 0.680245 | 0.682650 |
| E | 1 | SEX | 360 | catchup | C/catchup | 0.665039 | 0.668980 |
| E | 1 | SEX | 360 | pooled | C/catchup | 0.665039 | 0.668980 |
| E | 2 | RAC1P | 120 | independent | C/mlp_0 | 1.213488 | 1.218572 |
| E | 2 | RAC1P | 120 | catchup | C/catchup | 1.187212 | 1.189572 |
| E | 2 | RAC1P | 120 | pooled | C/catchup | 1.187212 | 1.189572 |
| E | 2 | RAC1P | 360 | independent | C/mlp_0 | 1.213488 | 1.218572 |
| E | 2 | RAC1P | 360 | catchup | C/catchup | 1.187212 | 1.189572 |
| E | 2 | RAC1P | 360 | pooled | C/catchup | 1.187212 | 1.189572 |
| E | 2 | SEX | 120 | independent | D/mlp_0 | 0.680054 | 0.680913 |
| E | 2 | SEX | 120 | catchup | D/catchup | 0.676122 | 0.679375 |
| E | 2 | SEX | 120 | pooled | D/mlp_0 | 0.680054 | 0.680913 |
| E | 2 | SEX | 360 | independent | D/mlp_0 | 0.680054 | 0.680913 |
| E | 2 | SEX | 360 | catchup | D/catchup | 0.676122 | 0.679375 |
| E | 2 | SEX | 360 | pooled | D/mlp_0 | 0.680054 | 0.680913 |
| S | 0 | RAC1P | 120 | independent | C/mlp_0 | 1.221718 | 1.204818 |
| S | 0 | RAC1P | 120 | catchup | D/catchup | 1.214920 | 1.199562 |
| S | 0 | RAC1P | 120 | pooled | D/catchup | 1.214920 | 1.199562 |
| S | 0 | RAC1P | 360 | independent | C/mlp_0 | 1.221718 | 1.204818 |
| S | 0 | RAC1P | 360 | catchup | D/catchup | 1.214920 | 1.199562 |
| S | 0 | RAC1P | 360 | pooled | D/catchup | 1.214920 | 1.199562 |
| S | 0 | SEX | 120 | independent | D/mlp_0 | 0.676931 | 0.677193 |
| S | 0 | SEX | 120 | catchup | D/catchup | 0.674107 | 0.678267 |
| S | 0 | SEX | 120 | pooled | D/catchup | 0.674107 | 0.678267 |
| S | 0 | SEX | 360 | independent | D/mlp_0 | 0.676931 | 0.677193 |
| S | 0 | SEX | 360 | catchup | D/catchup | 0.674107 | 0.678267 |
| S | 0 | SEX | 360 | pooled | D/catchup | 0.674107 | 0.678267 |
| S | 1 | RAC1P | 120 | independent | D/logistic | 1.260803 | 1.292277 |
| S | 1 | RAC1P | 120 | catchup | D/catchup | 1.248916 | 1.284898 |
| S | 1 | RAC1P | 120 | pooled | D/logistic | 1.260803 | 1.292277 |
| S | 1 | RAC1P | 360 | independent | D/logistic | 1.260803 | 1.292277 |
| S | 1 | RAC1P | 360 | catchup | D/catchup | 1.248916 | 1.284898 |
| S | 1 | RAC1P | 360 | pooled | D/logistic | 1.260803 | 1.292277 |
| S | 1 | SEX | 120 | independent | D/mlp_1 | 0.665832 | 0.670292 |
| S | 1 | SEX | 120 | catchup | D/catchup | 0.662474 | 0.667422 |
| S | 1 | SEX | 120 | pooled | D/mlp_1 | 0.665832 | 0.670292 |
| S | 1 | SEX | 360 | independent | D/mlp_1 | 0.665832 | 0.670292 |
| S | 1 | SEX | 360 | catchup | D/catchup | 0.662474 | 0.667422 |
| S | 1 | SEX | 360 | pooled | D/mlp_1 | 0.665832 | 0.670292 |
| S | 2 | RAC1P | 120 | independent | C/mlp_0 | 1.218560 | 1.219139 |
| S | 2 | RAC1P | 120 | catchup | D/catchup | 1.200172 | 1.203455 |
| S | 2 | RAC1P | 120 | pooled | D/catchup | 1.200172 | 1.203455 |
| S | 2 | RAC1P | 360 | independent | C/mlp_0 | 1.218560 | 1.219139 |
| S | 2 | RAC1P | 360 | catchup | D/catchup | 1.200172 | 1.203455 |
| S | 2 | RAC1P | 360 | pooled | D/catchup | 1.200172 | 1.203455 |
| S | 2 | SEX | 120 | independent | D/mlp_0 | 0.675239 | 0.674918 |
| S | 2 | SEX | 120 | catchup | C/catchup | 0.670920 | 0.673745 |
| S | 2 | SEX | 120 | pooled | D/mlp_0 | 0.675239 | 0.674918 |
| S | 2 | SEX | 360 | independent | D/mlp_0 | 0.675239 | 0.674918 |
| S | 2 | SEX | 360 | catchup | C/catchup | 0.670920 | 0.673745 |
| S | 2 | SEX | 360 | pooled | D/mlp_0 | 0.675239 | 0.674918 |
