"""Family B (Designer B) of the PCRL new-release-mechanisms campaign: channel-aware joint training of the two task predictors with a
SEX-targeted, Fisher-shaped, cross-recipient-coupled Gaussian logit channel ('FSC'). Card: results/<campaign>/cards/FB_FSC_v1.md.

    nrm.fb.channel   numpy only: centred-logit coordinates, per-person Fisher, channel geometry, per-person noise directions, keyed
                     one-draw-per-person emission (randomized), identity channel, deterministic mean-vector construction.
    nrm.fb.learner   torch: paired joint learners (task-only twin and channel-aware learners), CALIBRATION temperatures, the real fit
                     entry point (refuses before the registration receipt and without a committed card).
    nrm.fb.release   the nrm/deploy.py interface: predict(config_id, version, X, feature_names, draw=0, keys=None), model_paths(...).

Selftests (invented arrays only, no real data, no labels):
    python3 -m nrm.fb.channel selftest
    <repository>/.venv/bin/python -m nrm.fb.learner selftest      (also exercises nrm.fb.release.predict)
"""
