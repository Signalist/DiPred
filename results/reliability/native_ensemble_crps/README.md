# Native 8/32-path ensemble CRPS

These scores use actual independent-seed stochastic DiPred paths before any
online correction, with 8-noise equal to the first eight of the 32 seed range.
The raw ensemble means differ from the common zero-noise point used in the
XPF16 interval comparison. Do not label native ensemble CRPS as an online
calibrated interval score or directly compare it to a two-quantile head's
assumed Gaussian/Laplace completion.

`group_summary.json` and `trajectory_summary.json` retain warm-window averages
for each Re/channel and each trajectory/channel, respectively, for K=8,32 and
empirical/fair variants. Each aggregate saves A=mean |X-y| and B=0.5 × mean
ordered-pair |X-X'| under the relevant denominator; `CRPS=A-B`. Both standardized
and original-field terms are included. These processed data reproduce the
reported aggregate scores, but cannot support arbitrary new window-level,
pixelwise, spectral or joint-distribution analyses. Full window rows remain local.

`summary.json` includes test/development equal-Re/channel warm-window means.
`snapshot.npz` retains actual normalized values of 32 paths and target at 256
spacetime pixels for the first and last test trajectories at window0, plus each
pixel's 8/32 empirical/fair scores. Seeds, source hashes and event ordering are
in `protocol.json` and `events.jsonl`. The full 78-window ensembles (about
9.8 GB uncompressed at float32) and raw dataset are not stored in this Git repo;
regeneration needs the original data and frozen backbone. The model weights and
seeds are retained, and the research replay script is archived separately.

The empirical score uses denominator K²; the fair iid-member population estimator
uses K(K-1). These are distinct estimands. Positive scale standardization divides
CRPS by the train channel std; original physical units are not confirmed as SI.
Both variants are verified using explicit O(K²) pairwise distances on the snapshot.
All trajectory identities were previously used in research; results exploratory.
