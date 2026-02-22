# SoO Foundations

Algorithmic Foundations of Passive Radar & Signals of Opportunity (SoO)

Owner: Miguel Isarraraz  
Environment: Radioconda (GNU Radio + Python)

---

## Environment Setup

This project requires a working Radioconda installation.

Activate environment:

```powershell
conda activate C:\Users\general\radioconda
python -c "import gnuradio; print('OK')"
```

Choose one way to run package modules:

```powershell
# Option A: use source tree directly (current shell session)
$env:PYTHONPATH = "src"

# Option B: install editable package (recommended)
python -m pip install -e .
```

## Repository Layout

```text
SoO_Foundations/
|- src/
|  |- soo_foundations/
|  |  |- soo_sim.py              # Core synthetic signal + reflection model
|  |  |- soo_caf.py              # CAF computation utilities
|  |  |- s2_validate_sweep.py    # Stage 2 validation sweep engine
|  |  |- s4_tracking.py          # Stage 4 temporal tracking utilities (Kalman)
|  |
|  |- s1_gaussian.py             # Stage 1 Gaussian generator CLI
|  |- s1_chirp.py                # Stage 1 Chirp generator CLI
|  |- s2_caf.py                  # Single-run CAF visualizer CLI
|  |- plot_caf_heatmap.py        # Standard 2D CAF heatmap visualizer
|  |- plot_c64.py                # Time/Frequency signal plotting utility
|  |- plot_s2_validation.py      # Stage 2 validation report visualizer
|  |- plot_all_stages_report.py  # Unified Stage 1-3 presentation report
|  |- plot_isorange_ellipses.py  # Isorange ellipse plotter for one Tx/Rx pair
|  |- s3_monte_carlo_eval.py     # Stage 3 Monte Carlo metrics + plot
|  |- s3_multistatic_intersection_demo.py # 3-tower ellipse intersection with WLS
|  |- s4_tracking_demo.py        # Stage 4 temporal tracking demo + plot
|  |- s4_gdop_analysis.py        # Stage 4 GDOP baseline-geometry analysis
|
|- data/                         # Generated .c64 files (ignored by git)
|- results/                      # Sweep CSVs and report images
|- test/                         # Unit + integration tests
|- AGENT_MEMORY.md               # Repo-local AI collaboration preferences
|- BEGINNER_GUIDE.md             # New-user plain-language walkthrough
|- PROGRESS.md
|- README.md
```

New to this domain?
- Start with `BEGINNER_GUIDE.md` for a plain-language walkthrough of all stages and scripts.
- Assignment walkthrough notebook: `docs/SoO_Task3_Core_Algorithms_FY26_walkthrough.ipynb`

Graphics export note:
- Most plotting scripts save only when you pass `--save_png` (or `--out_png`).
- Save presentation artifacts to `results/` (examples below already do this).

## Run the Notebook (.ipynb)

Assignment notebook:
- `docs/SoO_Task3_Core_Algorithms_FY26_walkthrough.ipynb`

Launch Jupyter Notebook from this repo:

```powershell
conda activate C:\Users\general\radioconda
cd C:\swarm\repos\SoO_Foundations
$env:PYTHONPATH = "src"
python -m notebook
```

Then in the browser:
1. Open `docs/SoO_Task3_Core_Algorithms_FY26_walkthrough.ipynb`
2. Select kernel `Python (radioconda)` (or equivalent radioconda kernel)
3. Run `Kernel -> Restart & Run All`

If `notebook` module is missing:

```powershell
conda activate C:\Users\general\radioconda
python -m pip install notebook ipykernel
python -m ipykernel install --user --name radioconda --display-name "Python (radioconda)"
python -m notebook
```

## Stage 1: Generate and Inspect Signals

Generate Gaussian pair:

```powershell
python src\s1_gaussian.py --samp_rate 1e6 --seconds 3 --delay 2500 --doppler 30000 --attn 0.7 --snr_db 5 --seed 42 --out_dir data
```

Generate chirp pair:

```powershell
python src\s1_chirp.py --samp_rate 1e6 --f_start=-100e3 --f_end=100e3 --duration 2 --delay 1200 --doppler 50e3 --attn 0.9 --snr_db 10
```

Plot `.c64` time/frequency views:

```powershell
python src\plot_c64.py --original data\original_chirp.c64 --reflected data\reflected_chirp.c64 --samp_rate 1e6 --show_spectrogram
python src\plot_c64.py --original data\original_gaussian.c64 --reflected data\reflected_gaussian.c64 --samp_rate 1e6
```

All generated files are complex64 (.c64).
Approximate file size: `sample_rate * duration * 8 bytes`.

## Quick Demo Path

Run this sequence to produce presentation artifacts end-to-end:

```powershell
# 1) Generate chirp data
python src\s1_chirp.py --samp_rate 1e6 --f_start=-100e3 --f_end=100e3 --duration 2 --delay 1200 --doppler 50e3 --attn 0.9 --snr_db 10 --out_dir data

# 2) Render standard CAF heatmap
python src\plot_caf_heatmap.py --ref data\original_chirp.c64 --surv data\reflected_chirp.c64 --samp_rate 1e6 --delay_min 0 --delay_max 3000 --fd_min 0 --fd_max 80000 --fd_bins 161 --true_delay 1200 --true_doppler 50000 --show_contours --clip_db 50 --save_png results\caf_heatmap.png

# 3) Run Stage 2 sweep
python -m soo_foundations.s2_validate_sweep --center_on_truth --snr_db 60 --duration 0.2 --true_delay 1000 --fd_start 5000 --fd_stop 75000 --fd_cases 25 --delay_half_window 80 --fd_half_window 6000 --fd_bins 121 --pass_mode robust --out_csv results\stage2_sweep_present.csv --progress_every 0

# 4) Render validation report
python src\plot_s2_validation.py --csv results\stage2_sweep_present.csv --save_png results\stage2_validation_report.png --title "Stage 2 Delay-Doppler Validation" --no_show

# 5) Render one unified Stage 1-3 report
python src\plot_all_stages_report.py --original_c64 data\original_chirp.c64 --reflected_c64 data\reflected_chirp.c64 --stage2_csv results\stage2_sweep_present.csv --samp_rate 1e6 --mc_runs 200 --save_png results\all_stages_report.png --no_show
```

## Stage 2: CAF Check (Single Run)

Run CAF on one reference/surveillance pair and visualize peak:

```powershell
python src\s2_caf.py --ref data\original_chirp.c64 --surv data\reflected_chirp.c64 --samp_rate 1e6 --delay_min 0 --delay_max 3000 --fd_min 0 --fd_max 80000 --fd_bins 161 --true_delay 1200 --true_doppler 50000
```

Typical CAF heatmap (presentation style):

```powershell
python src\plot_caf_heatmap.py --ref data\original_chirp.c64 --surv data\reflected_chirp.c64 --samp_rate 1e6 --delay_min 0 --delay_max 3000 --fd_min 0 --fd_max 80000 --fd_bins 161 --true_delay 1200 --true_doppler 50000 --show_contours --clip_db 50 --save_png results\caf_heatmap.png
```

## Stage 2: Validation Sweep + Presentation Plot

Run a robust, truth-centered sweep and write CSV:

```powershell
python -m soo_foundations.s2_validate_sweep --center_on_truth --snr_db 60 --duration 0.2 --true_delay 1000 --fd_start 5000 --fd_stop 75000 --fd_cases 25 --delay_half_window 80 --fd_half_window 6000 --fd_bins 121 --pass_mode robust --out_csv results\stage2_sweep_present.csv --progress_every 0
```

Generate a presentation-ready validation report image from CSV:

```powershell
python src\plot_s2_validation.py --csv results\stage2_sweep_present.csv --save_png results\stage2_validation_report.png --title "Stage 2 Delay-Doppler Validation" --no_show
```

Optional: open the interactive figure (remove `--no_show`) to inspect points live.

## Stage 3: Positioning (Baseline)

Convert Stage 2 CSV into weighted Stage 3 measurements:

```powershell
python src\s3_measurements.py --csv results\stage2_sweep_present.csv --samp_rate 1e6 --scored_only --min_confidence 0.2
```

Run baseline 2D bistatic least-squares positioning demo:

```powershell
python src\s3_positioning_demo.py --seed 7 --noise_sigma_m 6
```

Run a multi-static (3 FM towers) ellipse intersection demo:

```powershell
python src\s3_multistatic_intersection_demo.py --seed 7 --noise_sigma_m 8 --txs_xy "0,0;16000,2500;-12000,9500" --rx_xy 3000,-2500 --target_xy 7000,5200 --save_png results\s3_multistatic_intersection.png
```

Plot isorange ellipses for a given transmitter/receiver pair:

```powershell
python src\plot_isorange_ellipses.py --tx_xy 0,0 --rx_xy 1000,0 --ranges_m 1200,1400,1700 --save_png results\isorange_ellipses.png
```

Or derive one isorange ellipse directly from a target point:

```powershell
python src\plot_isorange_ellipses.py --tx_xy 0,0 --rx_xy 1000,0 --target_xy 450,350
```

This demo prints:
- solver convergence status
- true vs estimated target coordinates
- position error (meters)
- residual RMS and weighted cost

Unified all-stage visualization:

```powershell
python src\plot_all_stages_report.py --original_c64 data\original_chirp.c64 --reflected_c64 data\reflected_chirp.c64 --stage2_csv results\stage2_sweep_present.csv --samp_rate 1e6 --mc_runs 200 --save_png results\all_stages_report.png --no_show
```

## Stage 3: Monte Carlo Evaluation (Step 3)

Run multi-geometry Monte Carlo evaluation and export metrics:

```powershell
python src\s3_monte_carlo_eval.py --noise_sigmas_m 1,2,4,8,12,16,20 --runs_per_case 300 --seed 11 --out_csv results\s3_monte_carlo_metrics.csv --out_png results\s3_monte_carlo_metrics.png --no_show
```

Outputs:
- `results/s3_monte_carlo_metrics.csv` (numeric metrics table)
- `results/s3_monte_carlo_metrics.png` (RMSE and convergence trend plot)

How to read the CSV columns:
- `rmse_m`: overall average position error magnitude (lower is better)
- `median_err_m`: typical error for that case (lower is better)
- `p95_err_m`: worst-case style error threshold for 95% of runs (lower is better)
- `convergence_rate_pct`: percentage of runs where solver converged (higher is better)

Expected trend checks:
- As `noise_sigma_m` increases, `rmse_m`, `median_err_m`, and `p95_err_m` should generally increase.
- Better geometry should produce lower errors than weak/biased geometry at the same noise.
- `convergence_rate_pct` may stay near 100% for moderate noise, then drop at high noise.

Recommendation for reporting:
- Use at least `runs_per_case >= 200` for presentation-quality stability.
- For final numbers, `runs_per_case` of `500` to `1000` is preferred.

## Stage 4: Temporal Tracking (New)

Run Stage 4 baseline tracking over noisy frame-by-frame positions:

```powershell
python src\s4_tracking_demo.py --seed 9 --steps 80 --dt_s 0.5 --noise_sigma_m 25 --sim_accel_sigma_mps2 0.8 --filter_accel_sigma_mps2 1.0 --save_png results\s4_tracking_demo.png --no_show
```

Outputs:
- `results/s4_tracking_demo.png` (trajectory + error trend plot)

What to check:
- `rmse_filtered_m` should usually be lower than `rmse_raw_m`.
- The filtered trajectory should look smoother than raw measurements.

Run Stage 4 GDOP / baseline-geometry analysis:

```powershell
python src\s4_gdop_analysis.py --runs_per_case 300 --noise_sigma_m 8 --baseline_tx_index 1 --baseline_distances_m 6000,4000,2500,1500,800,400,200 --out_csv results\s4_gdop_analysis.csv --save_png results\s4_gdop_analysis.png --no_show
```

Outputs:
- `results/s4_gdop_analysis.csv` (distance-to-baseline vs GDOP/error metrics)
- `results/s4_gdop_analysis.png` (geometry sweep + far/near intersection blur plots)

What to check:
- As distance-to-baseline decreases, `gdop` and `rmse_m` should generally increase.
- The near-baseline estimate cloud should look wider (more blurred) than far-baseline.

## Optional: Run Tests

```powershell
python -m unittest discover -s test -p "test_*.py"
```
