# F / P / PF / XF full-test CDF

CSV contains all seeds, raw, fixed-CQR and online calibrated 0.2% quantiles. See config.json for data split, grid and references. F/P/PF do not directly access X, although their DiPred-derived inputs depend on X. Test trajectories were previously used; comparisons are exploratory. Archived code assumes the original research workspace, cached inputs and source data; CSV/NPZ results can be read independently.

Replot without the original dataset: `import pandas as pd; import matplotlib.pyplot as plt; d=pd.read_csv("cdf_u_combined_online.csv"); d.plot(x="x"); plt.show()`.

Combined online curves use the same test trajectories, target-centred grid and original numerical scale. Heads use per-tail online offsets; noise uses its previously published 90% proxy-scale calibration, so calibration procedures differ. F/P/PF use zero normalized reference, whereas older X/XP/XPF use the last X frame; P/F themselves depend on X. The reference difference limits attribution to input slots alone.

Large CSV files are losslessly gzip-compressed; pandas.read_csv reads them directly.
