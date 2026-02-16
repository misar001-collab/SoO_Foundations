# SoO_Foundations – Task 3 Progress

Owner: Miguel Isarraraz  
Task: Algorithmic Foundations of Passive Radar & SoO Positioning (FY26)

## Current Status
- ✅ Stage 1 complete (Synthetic SoO Signal Generator)
- ⏳ Stage 2 next (CAF / matched filtering)

---

## Stage 1 – Synthetic SoO Signal Generator ✅

### What’s implemented
**Signals**
- Complex Gaussian noise
- Complex LFM chirp

**Reflected path model (matches Task 3 spec)**
- Delay: N samples
- Doppler: multiply by exp(j*2π*f_d*t)
- Attenuation: scale by α
- SNR scaling: sigma = sqrt(1/(2*10^(SNRdB/10)))

### Repo structure
- `src/s1_gaussian.py`  → generates original/reflected Gaussian (.c64)
- `src/s1_chirp.py`     → generates original/reflected chirp (.c64)
- `src/plot_c64.py`     → time-domain + FFT (+ optional spectrogram)

### Output artifacts
- `data/original_gaussian.c64`
- `data/reflected_gaussian.c64`
- `data/original_chirp.c64`
- `data/reflected_chirp.c64`

### Validation
- Delay visible in time-domain alignment (reflected starts later)
- Doppler visible in frequency-domain (FFT shift/offset)
- Chirp spectrogram shows the expected sweep and reflected offset

---

## Stage 2 – Signal Acquisition & Matched Filtering (CAF) ⏳
Next steps:
1. Implement CAF (delay x Doppler grid)
2. Plot CAF heatmap
3. Confirm peak equals injected (delay, doppler)
4. Optimize with FFT-based method

---

## Notes
- Development environment: radioconda (GNU Radio + Python)
- Use PowerShell-safe args for negatives: `--f_start=-100e3`
