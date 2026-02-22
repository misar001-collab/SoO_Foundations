# SoO Foundations - Task 3 Progress

Owner: Miguel Isarraraz  
Task: Algorithmic Foundations of Passive Radar and SoO Positioning (FY26)

## Current Status
- Completed: Stage 1 (Synthetic SoO Signal Generator)
- Completed: Stage 2 (CAF / matched filtering and validation)
- Completed: Stage 3 (Positioning from delay-Doppler measurements)
- Next: Stage 4 (Temporal tracking / frame-to-frame smoothing)
- Deliverable in progress: assignment walkthrough notebook scaffold added

---

## Stage 1 - Synthetic SoO Signal Generator (Complete)

Implemented:
- Complex Gaussian source generation
- Complex LFM chirp generation
- Reflected path model:
  - delay (samples)
  - Doppler rotation (`exp(j*2*pi*f_d*t)`)
  - attenuation scaling
  - SNR-based sigma scaling

Key scripts:
- `src/s1_gaussian.py`
- `src/s1_chirp.py`
- `src/plot_c64.py`
- `src/soo_foundations/soo_sim.py`

Artifacts:
- `data/original_gaussian.c64`
- `data/reflected_gaussian.c64`
- `data/original_chirp.c64`
- `data/reflected_chirp.c64`

Validation:
- Delay visible in time alignment
- Doppler visible in frequency-domain shift
- Chirp spectrogram behavior verified

---

## Stage 2 - Signal Acquisition and Matched Filtering (Complete)

Implemented:
- FFT-bank Cross Ambiguity Function (CAF)
- Delay-Doppler heatmap generation
- Global CAF peak detection
- Ground-truth error reporting
- Sweep-based validation with robust pass/fail logic
- Presentation report plotting from CSV outputs

Key scripts:
- `src/soo_foundations/soo_caf.py`
- `src/soo_foundations/s2_validate_sweep.py`
- `src/s2_caf.py`
- `src/plot_caf_heatmap.py`
- `src/plot_s2_validation.py`

Artifacts:
- `results/stage2_sweep_present.csv`
- `results/stage2_validation_report.png`
- `results/caf_heatmap.png`

Validation:
- CAF peak matches injected delay/Doppler under safe configuration
- Robust sweep run demonstrated `25/25` pass in scored cases

---

## Stage 3 - Positioning from Delay-Doppler (Next)

Planned objectives:
1. Define measurement model from Stage 2 outputs (Complete)
   - Delay and Doppler estimates
   - Confidence/uncertainty weighting
2. Implement baseline positioning solver (Complete - 2D bistatic WLS)
   - Least-squares estimator using known geometry
3. Add simulation-driven evaluation (Complete - Monte Carlo metrics + plots)
   - Monte Carlo over SNR, geometry, and measurement noise
4. Add temporal tracking option
   - Frame-to-frame smoothing (for moving targets)

Immediate next tasks:
- Integrate Stage 2 measurement ingestion directly into full geometry workflows
- Define Stage 3 pass metrics (RMSE, percentile error, convergence)
- Add Stage 3 acceptance thresholds and regression tests for CI

Step 1 implementation status:
- Added `src/soo_foundations/s3_positioning.py`
  - Stage 2 row -> Stage 3 measurement conversion
  - Delay/Doppler physical mapping (delay sec, bistatic range)
  - Uncertainty model and confidence scoring
- Added `src/s3_measurements.py`
  - CLI summary over Stage 2 CSV -> Stage 3 measurements

Step 2 implementation status:
- Added `src/soo_foundations/s3_solver.py`
  - 2D bistatic range prediction model
  - Weighted Gauss-Newton least-squares solver
  - Solver diagnostics (iterations, RMS residual, weighted cost, convergence reason)
- Added `src/s3_positioning_demo.py`
  - Synthetic geometry demo with controllable noise
  - Prints true vs estimated position and error metrics
- Added `src/s3_multistatic_intersection_demo.py`
  - Simulates 3 transmitter towers + 1 receiver (multi-static geometry)
  - Builds three noisy bistatic isorange ellipses
  - Solves target `(x,y)` using weighted least squares and plots ellipses/intersection
- Added `test/test_s3_solver_unit.py`
  - Geometry function unit test
  - Noiseless convergence test
  - Moderate-noise stability test

Step 3 implementation status:
- Added `src/soo_foundations/s3_evaluation.py`
  - Default geometry sets for evaluation
  - Monte Carlo runner over geometry x noise
  - Aggregated metrics: RMSE, median error, 95th percentile, convergence rate
- Added `src/s3_monte_carlo_eval.py`
  - CLI for Stage 3 Monte Carlo sweeps
  - CSV export for metrics
  - PNG plot export for presentations
- Added `test/test_s3_evaluation_unit.py`
  - Noise parser test
  - Output schema/metric bounds test
  - RMSE trend test with increasing noise

---

## Stage 4 - Temporal Tracking (Started)

Objective:
- Smooth noisy frame-by-frame position estimates from Stage 3.

Step 1 implementation status:
- Added `src/soo_foundations/s4_tracking.py`
  - 2D constant-velocity Kalman tracking model
  - Configurable process and measurement noise
  - Per-step tracking outputs (measured, predicted, filtered, innovation)
- Added `src/s4_tracking_demo.py`
  - Simulated moving-target trajectory with noisy position measurements
  - Runs tracker and reports RMSE before/after filtering
  - Exports visualization to `results/s4_tracking_demo.png`

Step 2 implementation status:
- Added `src/s4_gdop_analysis.py`
  - Multi-static GDOP geometry analysis with 3 towers + 1 receiver
  - Sweeps target distance-to-baseline and runs Monte Carlo solves
  - Exports metrics CSV (`results/s4_gdop_analysis.csv`)
  - Exports figure with near/far blur intuition (`results/s4_gdop_analysis.png`)
- Added Stage 4 unit tests:
  - `test/test_s4_tracking_unit.py`
  - `test/test_s4_gdop_analysis_unit.py`
  - `test/test_s4_cli_unit.py`
  - Includes fixed-seed GDOP regression (near-baseline RMSE degradation)
  - Includes CLI output-file smoke tests for `s4_tracking_demo.py` and `s4_gdop_analysis.py`
  - Verified via `python -m unittest discover -s test -p \"test_s4*_unit.py\"` (10 tests, pass)

Deliverables status update:
- Added `docs/SoO_Task3_Core_Algorithms_FY26_walkthrough.ipynb`
  - Stage 1-4 runnable walkthrough scaffold
  - Includes synthetic SNR benchmark table cell (`10`, `0`, `-10` dB)
