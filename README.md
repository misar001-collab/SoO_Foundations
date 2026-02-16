@"
# Signals of Opportunity Passive Radar



## Quickstart (Stage 1)


### Generate Chirp original/reflected
python src\s1_gaussian.py --seconds 2

### Generate Chirp original/reflected
python src\s1_chirp.py --samp_rate 1e6 --f_start=-100e3 --f_end=100e3 --duration 2 --delay 1200 --doppler 50e3 --attn 0.9 --snr_db 10


### Plot time + frequency for chirp
python src\plot_c64.py --original data\original_chirp.c64 --reflected data\reflected_chirp.c64 --samp_rate 1e6 --show_spectrogram
