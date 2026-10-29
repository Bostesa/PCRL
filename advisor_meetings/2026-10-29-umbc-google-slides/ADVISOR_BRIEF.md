# What we need to agree with Dr. Yus

Before discussing an algorithm, we need to explain who the attacker is, why it receives the information in our tests, and what extra inference we want to prevent.

## Recommended attacker

A person or organisation is allowed to receive information from a prediction service for a stated task. It then tries to use that information to infer an attribute the policy says it should not use. For example, a company might receive income predictions and try to infer sex. This is an illustration, not a service we have documented.

For each target person, the attacker receives only the information the service agrees to give it. It does not get the original record, model weights or repeated queries. It also receives other people’s outputs paired with known sensitive labels. It uses these examples to train and choose a classifier for predicting the sensitive attribute. The availability of these labelled outputs is an explicit benchmark assumption. We have not established how a real recipient would obtain them.

Keep two recipients pooling their outputs as an optional extension. That test grants exact matching of the same person across both outputs. It does not measure how the parties obtain both outputs or achieve that match.

## How this relates to prior work

Attribute-inference attacks already learn from labelled examples. LEACE promises protection against linear classifiers, so nonlinear recovery does not contradict its guarantee. FARE covers classifiers using its cell output. Giving the attacker an additional untreated output changes the information available to it. Our contribution is the comparison of released information and usefulness on the same stored models, not a new kind of attacker.

## What the existing tests establish

The tests separate fitting the attack, choosing it on validation examples and scoring it on another group. However, FARE and noise choices also used attacker-validation feedback. Some usefulness measurements concern a separate task classifier rather than the model providing the output. All scoring records had been used in earlier analyses. These are development results against the tested classifiers, not a guarantee against every attack. No successful new algorithm has been established.

## Three meeting questions

1. Is this recipient and its labelled-example access the right primary attacker, even if every method resists it?
2. Does the permitted service need features, scores or only decisions? Should pooling two recipients remain a separate extension?
3. With those assumptions stated, what methodological limits must the paper retain before we discuss a proposed algorithm and then paper structure?
