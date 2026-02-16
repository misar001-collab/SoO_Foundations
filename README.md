# SoO Foundations

Algorithmic Foundations of Passive Radar & Signals of Opportunity (SoO)

Owner: Miguel Isarraraz  
Environment: Radioconda (GNU Radio + Python)

---

# Environment Setup

This project requires a working Radioconda installation.

Activate environment:

```powershell
conda activate C:\Users\general\radioconda
python -c "import gnuradio; print('OK')"
#######################################################################################################
SoO_Foundations/
│
├── src/
│   ├── soo_sim.py          # Core synthetic signal functions
│   ├── s1_gaussian.py      # CLI wrapper (Gaussian pair)
│   ├── s1_chirp.py         # CLI wrapper (Chirp pair)
│   ├── plot_c64.py         # Time/Frequency plotting utility
│
├── data/                   # Generated .c64 files (ignored by git)
├── PROGRESS.md             # Task progress tracking
└── README.md


#########Stage1 Start#############################################################################

S1_Gaussian:
python src\s1_gaussian.py --samp_rate 1e6 --seconds 3 --delay 2500 --doppler 30000 --attn 0.7 --snr_db 5 --seed 42 --out_dir data

S1_Chirp:
python src\s1_chirp.py --samp_rate 1e6 --f_start=-100e3 --f_end=100e3 --duration 2 --delay 1200 --doppler 50e3 --attn 0.9 --snr_db 10

Plot_C64:
python src\plot_c64.py --original data\original_chirp.c64 --reflected data\reflected_chirp.c64 --samp_rate 1e6 --show_spectrogram

All generated files are complex64 (.c64)
File size ≈ sample_rate × duration × 8 bytes
.c64 files are excluded from Git

###################END of Stage 1#################################################################