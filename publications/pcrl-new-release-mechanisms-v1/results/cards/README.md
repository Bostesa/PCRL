# Mechanism cards

- **Commit before fit.** A card governs every fit of its family or control. It is committed here before the first fit it governs.
- **Enforced at load time.** The access layer refuses a 'fit' or 'calibrate' label read without a committed, unchanged card, and logs
  the card's path and SHA-256 with every read.
- **Revisions.** A revision is a new card file (`<FAMILY>_CARD_v<k>.md`) recording the coordinator's and the statistical reviewer's
  approval before its fit. Earlier cards are never edited.

## Required sections

1. **Operation:** the exact operation that changes the transmitted output (raw fields -> income vector, occupation vector; decisions
   = argmax). For a randomized release, also the channel law, the noise coupling between the two recipients, the seed handling,
   the number R of Monte Carlo repetitions and the reader sampling access for auxiliary people.
2. **Closest prior PCRL attempt** (folder / file), what is substantively different, and which measured bottleneck it addresses.
3. **Why it could improve protection** without repeating the observed utility cost.
4. **Task-only twin.** Same architecture, adequate capacity, equal data and tuning effort.
5. **Simplest control(s)** that could explain a positive result: at least one SEX-blind control path of comparable utility cost;
   the relevant published objective where one exists.
6. **Falsifying outcome:** the outcome that would falsify the rationale.
7. **Fitting plan:**
   - roles and labels read;
   - full-pool or internal folds (household-grouped; nested or non-nested);
   - actual training sizes;
   - calibration role;
   - seeds per version;
   - the predeclared strength path.
8. **Readers needed** to interpret the release (finite cells: LOOKUP; randomized: one-draw channel-aware readers).
9. **Budget:** configurations requested (each (config, version) counts) and CPU estimate.
10. **Revision plan:** which evidence triggers which revision, replacement or stop.
