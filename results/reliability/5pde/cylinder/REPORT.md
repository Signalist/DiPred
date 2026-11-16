# Five-PDE reliability experiment: cylinder

Real-observation dataset only. Train/development/calibration/test trajectories: train=50, development=11, calibration=11, test=20.
Evaluated 20 test trajectories, 197 valid non-cold-start windows per trajectory, 20 future frames per window.
One scenario-specific diffusion predictor is trained from scratch on training trajectories, then frozen. X, XP and XPF use the same quantile head, X-last-frame reference, training data, batch sequence, optimizer and three matched initialization seeds.
Noise baselines are actual four-step diffusion trajectories; 8 and 32 paths are prefixes of 256 independently seeded paths. The same empirical quantile interpolation, fixed CQR and causal online calibration are applied, and finite-sample tail quantiles are inherently coarse.
The shared head uses 12 spatiotemporal input channels: X=2, P=2 and an 8-channel projection of the 48-channel F; this is the fair shared-head design, not the historical XPF16 head. Missing P/F slots are masked.
P and F are derived only from the observed history. F is the final hidden feature of the fourth actual denoising call. For combustion the unused second padded channel is excluded from losses and metrics.
Original stored fields are strided to 32×64; physical coordinate spacing and control parameters are not provided as model inputs. The two source axes may be downsampled by different factors.
Raw is the primary comparison; split CQR and causal online calibration are auxiliary. Calibration pools correlated space/time points and therefore does not by itself guarantee trajectory-level or distribution-shift coverage.
Raw noise CDFs use the actual empirical samples. Fixed/online noise CDFs use calibrated quantile functions and are not claimed to be calibrated random ensembles.
Each trajectory window is issued before its complete target is revealed. The starting window is used only to initialize the online state. Raw quantiles are monotone by construction; calibrated quantiles are monotonically rearranged before scoring.
CRPS uses the same 499 levels at .002 spacing with trapezoidal 2×pinball integration and clamped tails. CDF x axes are in original data units, the same for all arms and calibration methods within the scenario.
Per-window and 20-column per-frame CSVs are split by model/seed for transfer size; *_index.csv lists every part. All valid windows and field locations are represented.
Bootstrap resamples trajectories; per-window intervals are unadjusted for multiple comparisons, and shared physical-parameter groups and spatial/time dependence limit their interpretation. Earlier project use of these trajectories is not fully known, so the test results are exploratory.
The source archive has real and simulated domains; only the real domain is used here. In FSI, six official entries overlap remain/out-of-distribution lists and six IDs are unlisted; all such IDs are reserved for test.

Audit passed: True.

| Method | Seed | Channel | Raw CRPS | Raw 90% coverage | Raw interval score |
|---|---:|---|---:|---:|---:|
| X | 42 | u | 0.0040729032 | 0.866203 | 0.04026575 |
| X | 42 | v | 0.0084084888 | 0.802830 | 0.07553536 |
| XP | 42 | u | 0.0036948238 | 0.854748 | 0.036166559 |
| XP | 42 | v | 0.0073546803 | 0.794536 | 0.06534607 |
| XPF | 42 | u | 0.0033132758 | 0.850581 | 0.032632751 |
| XPF | 42 | v | 0.0056957628 | 0.780033 | 0.052960844 |
| X | 123 | u | 0.0041017096 | 0.822700 | 0.041053272 |
| X | 123 | v | 0.0083436486 | 0.803624 | 0.07473536 |
| XP | 123 | u | 0.0037225477 | 0.816581 | 0.0374941 |
| XP | 123 | v | 0.0073700394 | 0.801141 | 0.065501244 |
| XPF | 123 | u | 0.0032947189 | 0.814173 | 0.033193694 |
| XPF | 123 | v | 0.0057157214 | 0.787942 | 0.053280371 |
| X | 2026 | u | 0.0041092033 | 0.820735 | 0.042552756 |
| X | 2026 | v | 0.0084077596 | 0.809744 | 0.076131644 |
| XP | 2026 | u | 0.0037376431 | 0.804067 | 0.038200195 |
| XP | 2026 | v | 0.0074094918 | 0.802648 | 0.066073761 |
| XPF | 2026 | u | 0.0033031999 | 0.796516 | 0.033695327 |
| XPF | 2026 | v | 0.0057391736 | 0.789468 | 0.05339662 |
| 8-noise | 500000 | u | 0.0035271196 | 0.698733 | 0.048106031 |
| 8-noise | 500000 | v | 0.0061666619 | 0.663181 | 0.078188413 |
| 32-noise | 500000 | u | 0.0033628303 | 0.791862 | 0.042913663 |
| 32-noise | 500000 | v | 0.0058323484 | 0.764567 | 0.067278375 |
| 256-noise | 500000 | u | 0.0033073537 | 0.817269 | 0.041220607 |
| 256-noise | 500000 | v | 0.0057238285 | 0.795520 | 0.063827785 |
