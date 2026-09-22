# Randomness and loss

**Primary per-person loss.** For a frozen fitted predictor f and person i,

  L_i = Σ_z Q(z | x_i) · (−log max(f(H_i, z)_{y_i}, 1e-9))

This is computed exactly, in chunks of 256 people over all actions (predecessor `expected_token_loss`).

- It averages losses. It never takes the loss of Σ_z Q·f.
- It is the expected loss of one sampled token. It is not the loss of a recipient who observes every token.
- Q(x_i) is used only by this offline integral and never enters an attacker's input.
- Integrating over tokens removes token noise only. It does not remove uncertainty about households or the
  fitted models.

**How each family is evaluated.**

- Deterministic releases (H, J, C, E, S, D17, D33) are the one-action case.
- RR75 uses `0.75·Q_D17(x) + 0.25/17`.
- W75 uses `[0.75·Q_D17(x), 0.25 withheld]`; the withheld symbol decodes to b(H_A).

**Attacker training on stochastic inputs.** This is the predecessor's validated implementation, unchanged:

- Exact person-token expansion with person weights conserved (logistic, exact HistGB, MLP minibatches).
  An objective counts original people.
- Two sampled-token HistGB candidates that use one fixed categorical draw per person (seed 20260921 + role
  seed). The draws are saved privately.
- Validation always uses the exact expectation. Attackers are never refitted after final outcomes.

**One-token replay fixture (secondary).** A persistent private token per person is drawn as follows:

- u_i = HMAC-SHA256(secret, person_id) / 2^256.
- The token is the inverse CDF of the Q row at u_i.
- Repeated requests return the same token. A test checks this.
- For a representative unit (Q, anchor 0, A/SEX, final pool), the realized loss with one draw per person
  is averaged over 200 independent secrets. Each secret stands for a different population's coins, not a
  redraw of the same person. The average must agree with the exact expected loss within Monte Carlo error.
- Neither the tokens nor the secret enter scoring, and the replications do not add to the sample size.
