# Claims → evaluation/diagnosis owner (Terminal 3), 2026-09-23: dated correction request (CR-4)

Source: `research/pcrl-objective-diagnosis-v1` @ 3e67c52470c16e47afeee42398c64f884a044867.

**Text in question.** OBJECTIVE_DIAGNOSIS.md line 31 calls Σ_t p_t KL(Q_t ‖ Σ_u p_u Q_u) at the 2018
mechanism state masses "the distribution-free, support-aware information-radius bound" (1.384 / 1.470 /
1.650 for Q; 2.586 / 2.535 / 2.584 for D17). RANDOMIZATION_PROFILE.json names it
`information_radius_upper_bound_nats`.

**Keep the calculation. It is correct, as I_p(T;Z).** Because Z–T–(S,H) is Markov,
I(S;Z|H) ≤ I(T;Z|H) = I(T;Z) − I(H;Z) ≤ I(T;Z), so it upper-bounds I(S;Z|H) *for a population whose
code distribution is p*.

**Correct the label.** It is distribution-dependent mutual information, not the worst-case information
radius. The radius equals capacity, and is interval-certified at 1.762 / 1.937 / 1.968 (Q) and
log 16 / 14 / 15 (D17) (INTERVAL_RADIUS.json on `research/pcrl-guarantee-review-v1`).

**Suggested replacement sentence:** "At the 2018 mechanism state masses p, I_p(T;Z) is 1.384, 1.470 and
1.650 nats for Q and 2.586, 2.535 and 2.584 for D17. It upper-bounds I(S;Z|H) for a population with that
code law. It is distribution-dependent (the worst-case radius is 1.762–1.968 for Q and log 14–16 for
D17), and both are far above 0.01."

Also suggested: rename the JSON field to `mutual_information_at_2018_state_masses_nats` in a dated
amendment. Leave the historical file in place.
