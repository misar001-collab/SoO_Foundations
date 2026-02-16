#!/usr/bin/env python3

import argparse
import math
import time
import signal
from pathlib import Path
from gnuradio import gr, blocks, analog


def snr_to_sigma(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    return math.sqrt(1.0 / (2.0 * snr_linear))


class S1Gaussian(gr.top_block):
    def __init__(self, samp_rate, doppler, delay, attn, snr_db, seconds):

        super().__init__("S1_Gaussian")

        phase_inc = 2 * math.pi * doppler / samp_rate
        sigma = snr_to_sigma(snr_db)

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        original_path = data_dir / "original_gaussian.c64"
        reflected_path = data_dir / "reflected_gaussian.c64"

        noise = analog.noise_source_c(analog.GR_GAUSSIAN, 1.0, 0)

        delay_block = blocks.delay(gr.sizeof_gr_complex, delay)
        rot = blocks.rotator_cc(phase_inc)
        attn_block = blocks.multiply_const_cc(attn)
        sigma_block = blocks.multiply_const_cc(sigma)

        throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        sink_orig = blocks.file_sink(gr.sizeof_gr_complex, str(original_path), False)
        sink_refl = blocks.file_sink(gr.sizeof_gr_complex, str(reflected_path), False)

        self.connect(noise, throttle, sink_orig)

        self.connect(noise, delay_block, rot, attn_block, sigma_block, sink_refl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samp_rate", type=float, default=1e6)
    parser.add_argument("--doppler", type=float, default=50000)
    parser.add_argument("--delay", type=int, default=1200)
    parser.add_argument("--attn", type=float, default=0.9)
    parser.add_argument("--snr_db", type=float, default=10)
    parser.add_argument("--seconds", type=float, default=2)

    args = parser.parse_args()

    tb = S1Gaussian(
        args.samp_rate,
        args.doppler,
        args.delay,
        args.attn,
        args.snr_db,
        args.seconds
    )

    tb.start()
    time.sleep(args.seconds)
    tb.stop()
    tb.wait()


if __name__ == "__main__":
    main()
