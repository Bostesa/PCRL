# R15 review: sealed fixed-reference attack correction

Status: **SCOPED STATIC GO** for the dated 12-attempt correction. Real attack outcomes remain pending.

Correction SHA-256 `5cfcf570e27d623625db30f2be61727934f1290d7cef9fdb33a8a38fa146d613`, committed at `b918648abe20eca54446f56aa7815f280e13080e`.

I compared every physical R1 attempt to the unchanged original producer unit plan. All 12 map one-to-one to the original logical arm and seed, with the same release/Q dependencies and reader counts. Eight finite attempts retain 33 fits each; four continuous attempts retain 24 fits each, totaling 360 planned fits. Every attempt has a 90-second CPU/wall cap and 3-GiB memory cap; the correction declares one heavy slot. The producer plan digest, R03 queue proposal hash, and four new source hashes match the live files.

The two original U0/P* S0 attempts remain failed, with their original queue receipt hashes, return code 1, and empty outputs preserved. Ten original units remain unqueued. The sealed public technical-failure custody and queued data-free test receipt match their hashes; the latter reports 12/12 synthetic tests passed. This was a metadata-only review, with no Adult arrays opened. It does not convert technical failures into results or establish a real attack outcome.
