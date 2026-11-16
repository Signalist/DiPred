# Legacy 16-channel input and masking

All channels are presented in the order used by the original 16-input Conv3D.
The first six are normalized using the frozen legacy base-feature statistics,
then clamped to [-10, 10]. The denoising feature is normalized and clamped using
its frozen legacy statistics before the learned projection. Mask value zero is
the neutral value in these normalized slots, not necessarily physical zero.

| Head channel | Actual formula / origin | Required direct input |
|---|---|---|
| 0, 1 | Per-channel standard deviation of the normalized 20-frame history, repeated over future time | X |
| 2, 3 | Norm of the spatial grid gradient of normalized P | P |
| 4, 5 | Absolute normalized P time difference: first future frame is `abs(P[0]-X[-1])`; later frames are `abs(P[t]-P[t-1])` | P and X at t=0; P only for t>0 |
| 6–13 | Last of the four frozen DiPred `up2` denoising feature maps (48 channels), cast through float16 as in the legacy extractor, normalized, then the original trainable 1×1 projection 48→160 reshaped to 8×20 | F |
| 14, 15 | x and y coordinate grids, each spanning [-1, 1] | Common geometry |

When X is absent, channels 0–1 and the first future frame of channels 4–5
are set to normalized zero. When P is absent, channels 2–5 are set to zero.
When F is absent, channels 6–13 are set to zero **after** projection, preventing
its bias from entering the head. Coordinates remain common. There are no
additional learned adapters. The common output anchor is P in every arm:
`Q(Y)=P+Q(Y-P)`. F and P depend indirectly on X through the frozen predictor;
therefore F-only is not a claim of an end-to-end X-free or P-free system.
No future ground truth is used to construct the features or issue an interval.
