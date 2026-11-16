# Five-PDE reliability experiment: controlled_cylinder

Real-observation dataset only. Train/development/calibration/test trajectories: train=54, development=11, calibration=11, test=20.
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
| X | 42 | u | 0.0067454583 | 0.853560 | 0.063406413 |
| X | 42 | v | 0.005752549 | 0.805174 | 0.051376426 |
| XP | 42 | u | 0.0057829077 | 0.852772 | 0.054282149 |
| XP | 42 | v | 0.0048584194 | 0.812352 | 0.042481512 |
| XPF | 42 | u | 0.0049943762 | 0.847487 | 0.046756736 |
| XPF | 42 | v | 0.0037740082 | 0.789282 | 0.034486295 |
| X | 123 | u | 0.0067619919 | 0.840023 | 0.063897418 |
| X | 123 | v | 0.0057126196 | 0.815981 | 0.050929604 |
| XP | 123 | u | 0.0057875733 | 0.832339 | 0.054976516 |
| XP | 123 | v | 0.0048595543 | 0.810431 | 0.042617393 |
| XPF | 123 | u | 0.0050533303 | 0.840777 | 0.048491597 |
| XPF | 123 | v | 0.0037668762 | 0.796327 | 0.034617135 |
| X | 2026 | u | 0.0067343258 | 0.839843 | 0.063558023 |
| X | 2026 | v | 0.0057581443 | 0.822606 | 0.051202974 |
| XP | 2026 | u | 0.0058128628 | 0.840753 | 0.055080711 |
| XP | 2026 | v | 0.0048911654 | 0.821858 | 0.042622495 |
| XPF | 2026 | u | 0.0049750859 | 0.822240 | 0.047160398 |
| XPF | 2026 | v | 0.0037975216 | 0.798868 | 0.034692951 |
| 8-noise | 500000 | u | 0.0052491964 | 0.727665 | 0.060248172 |
| 8-noise | 500000 | v | 0.0040239495 | 0.658093 | 0.046540545 |
| 32-noise | 500000 | u | 0.0049205564 | 0.826701 | 0.051418582 |
| 32-noise | 500000 | v | 0.0037762416 | 0.766825 | 0.038773083 |
| 256-noise | 500000 | u | 0.004814905 | 0.854984 | 0.048904197 |
| 256-noise | 500000 | v | 0.0036974584 | 0.801467 | 0.036419923 |
