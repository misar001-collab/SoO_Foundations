# SoO Foundations Beginner Guide

This guide explains the project in plain language for someone new to passive radar and Signals of Opportunity (SoO).

## 1) Big Picture

This project is a 3-stage pipeline:

1. Stage 1: Create synthetic signals that act like what a receiver would capture.
2. Stage 2: Compare those signals to estimate delay and Doppler (how late and how fast).
3. Stage 3: Use those estimates to infer target position and evaluate solver quality.

If you remember only one thing:
- Stage 1 makes data.
- Stage 2 extracts measurements.
- Stage 3 turns measurements into position.

## 2) Core Concepts (No Prior Background Needed)

- `Reference signal`: the direct signal path (what was transmitted).
- `Surveillance signal`: the reflected path from a target (arrives later, shifted in frequency).
- `Delay`: how many samples later the reflection appears (related to extra path length).
- `Doppler`: frequency shift due to relative motion (related to range rate).
- `CAF (Cross Ambiguity Function)`: a 2D search over delay and Doppler to find the best match.
- `Isorange ellipse`: in bistatic geometry, all target points with the same total path length `|p-tx| + |p-rx|`.

## 3) Stage 1 - Signal Generation and Inspection

Goal: generate known test data where truth is controlled.

### `src/s1_gaussian.py`
- Creates an original random complex Gaussian signal.
- Creates a reflected version by applying delay, Doppler, attenuation, and SNR scaling.
- Writes outputs as `.c64` files (complex64 IQ samples).

Why it matters:
- Gives controlled test cases so later stages can be validated against known settings.

### `src/s1_chirp.py`
- Same idea as `s1_gaussian.py`, but uses an LFM chirp waveform.
- Also writes original/reflected `.c64` files.

Why it matters:
- Chirps are structured signals and often easier to inspect visually.

### `src/plot_c64.py`
- Plots time-domain and FFT magnitude views of `.c64` files.
- Optional spectrogram mode.

Why it matters:
- Quick sanity check that delay/Doppler effects are visible before running CAF.

Expected Stage 1 outputs:
- `data/original_*.c64`
- `data/reflected_*.c64`

## 4) Stage 2 - Delay-Doppler Estimation (CAF)

Goal: recover delay/Doppler from Stage 1 signal pairs.

### `src/soo_foundations/soo_caf.py` (library)
- Implements CAF computation utilities.
- Provides peak-finding helpers.

Why it matters:
- This is the estimation engine used by the Stage 2 scripts.

### `src/s2_caf.py`
- Runs one CAF estimate on one signal pair.
- Prints estimated peak delay and Doppler.
- Can compare estimates against provided truth values.
- Shows a CAF heatmap.

Why it matters:
- Fast single-run diagnostic for "did estimation work on this case?"

### `src/plot_caf_heatmap.py`
- Creates a presentation-friendly CAF heatmap.
- Highlights strongest peak and can overlay truth markers.

### `src/soo_foundations/s2_validate_sweep.py` (library/CLI module)
- Runs many cases (parameter sweep) instead of one.
- Writes a CSV with per-case metrics and pass/fail style fields.

### `src/plot_s2_validation.py`
- Reads Stage 2 sweep CSV.
- Produces summary visuals showing performance trends.

Expected Stage 2 outputs:
- `results/stage2_sweep_present.csv`
- `results/caf_heatmap.png`
- `results/stage2_validation_report.png`

How to interpret Stage 2 at a high level:
- Smaller delay error is better.
- Smaller Doppler error is better.
- More pass cases is better.
- Stable behavior across sweep settings is better than one isolated good run.

## 5) Stage 3 - Positioning from Measurements

Goal: convert delay/Doppler measurements into target position estimates.

### `src/soo_foundations/s3_positioning.py` (library)
- Converts Stage 2 CSV rows into Stage 3 measurement objects.
- Handles confidence/uncertainty modeling for weighting.

### `src/soo_foundations/s3_solver.py` (library)
- Implements a 2D bistatic weighted least-squares solver.
- Returns convergence and diagnostic details.

### `src/s3_measurements.py`
- Summarizes and filters Stage 2-to-Stage 3 measurements.
- Useful for checking data quality before solving.

### `src/s3_positioning_demo.py`
- Runs a synthetic 2D positioning example.
- Prints true vs estimated location and error stats.

### `src/s3_multistatic_intersection_demo.py`
- Simulates one receiver and three transmitter towers.
- Builds three bistatic ellipse constraints (one per tower).
- Solves for the best-fit `(x, y)` intersection with weighted least squares.
- Plots towers, receiver, ellipses, true target, and estimated target.

### `src/plot_isorange_ellipses.py`
- Plots one or more isorange ellipses for a single Tx/Rx pair.
- Useful for geometry intuition and presentations.

### `src/soo_foundations/s3_evaluation.py` + `src/s3_monte_carlo_eval.py`
- Runs Monte Carlo studies across noise levels/geometries.
- Exports CSV and plot with aggregate metrics.

Expected Stage 3 outputs:
- `results/s3_multistatic_intersection.png` (when `--save_png` is used)
- `results/isorange_ellipses.png` (when `--save_png` is used)
- `results/s3_monte_carlo_metrics.csv`
- `results/s3_monte_carlo_metrics.png`

How to interpret Stage 3 metrics:
- `rmse_m`: lower is better.
- `median_err_m`: lower is better (typical case).
- `p95_err_m`: lower is better (near worst-case tail).
- `convergence_rate_pct`: higher is better.

## 6) Stage 4 - Temporal Tracking (New)

Goal: smooth frame-to-frame position estimates from Stage 3.

### `src/soo_foundations/s4_tracking.py` (library)
- Implements a 2D constant-velocity Kalman tracker.
- Uses model + measurement uncertainty to balance smoothness and responsiveness.

### `src/s4_tracking_demo.py`
- Simulates a moving target with noisy position measurements.
- Runs the Stage 4 tracker and reports:
  - `rmse_raw_m` (before filtering)
  - `rmse_filtered_m` (after filtering)
- Exports a figure with trajectory and error trends.

### `src/s4_gdop_analysis.py`
- Analyzes how tower/receiver geometry affects accuracy (GDOP intuition).
- Moves the target toward the selected tower-receiver baseline.
- For each distance-to-baseline, runs Monte Carlo position solves and reports:
  - `gdop` (geometry-only sensitivity)
  - `rmse_m`, `median_err_m`, `p95_err_m` (empirical error)
  - `blur_area_m2` (estimate-cloud spread proxy)
- Exports a plot comparing far-baseline vs near-baseline estimate clouds.

Expected Stage 4 output:
- `results/s4_tracking_demo.png`
- `results/s4_gdop_analysis.csv`
- `results/s4_gdop_analysis.png`

## 7) Graphics Export Rules

Most plotting scripts save figures when you pass `--save_png` (or `--out_png`), and outputs should be placed in `results/`.

Examples:

```powershell
python src\plot_caf_heatmap.py --ref data\original_chirp.c64 --surv data\reflected_chirp.c64 --samp_rate 1e6 --delay_min 0 --delay_max 3000 --fd_min 0 --fd_max 80000 --fd_bins 161 --save_png results\caf_heatmap.png
python src\plot_isorange_ellipses.py --tx_xy 0,0 --rx_xy 1000,0 --ranges_m 1200,1400,1700 --save_png results\isorange_ellipses.png
python src\s3_multistatic_intersection_demo.py --save_png results\s3_multistatic_intersection.png
python src\s4_tracking_demo.py --save_png results\s4_tracking_demo.png
python src\s4_gdop_analysis.py --out_csv results\s4_gdop_analysis.csv --save_png results\s4_gdop_analysis.png
```

## 8) Suggested Learning Path (New User)

1. Run Stage 1 once and inspect `plot_c64` output.
2. Run `src/s2_caf.py` and compare estimated values against truth.
3. Run Stage 2 sweep and open the validation report.
4. Run `src/plot_isorange_ellipses.py` to build geometric intuition.
5. Run `src/s3_multistatic_intersection_demo.py` to see ellipse intersection solving.
6. Run `src/s3_monte_carlo_eval.py` and inspect trend metrics.
7. Run `src/s4_tracking_demo.py` to see frame-to-frame smoothing.
8. Run `src/s4_gdop_analysis.py` to see baseline-geometry degradation (GDOP).

This order mirrors how the project is built: data -> measurements -> position.

## 9) Common Beginner Mistakes

- Mixing reference/surveillance files from different runs.
- Using too few samples for stable CAF behavior.
- Assuming one good case means robust performance.
- Ignoring confidence/quality filters before Stage 3 solving.
- Comparing values without tracking units (samples, Hz, meters).
- Forgetting that many scripts only save plots when `--save_png` is provided.
- Assuming one noisy frame is meaningful without temporal tracking context.
- Ignoring geometry effects: near-baseline targets are often less observable.

## 10) Practical Rule of Thumb

Before trusting a result:
1. Visualize signals (Stage 1).
2. Verify delay/Doppler recovery on known truth (Stage 2).
3. Check multi-static solve and Monte Carlo trends (Stage 3).
4. Check whether tracking reduces RMSE and jitter (Stage 4).
5. Check GDOP and near-baseline blur behavior before trusting geometry-sensitive cases.

If all three are consistent, you are likely in a good operating regime.
