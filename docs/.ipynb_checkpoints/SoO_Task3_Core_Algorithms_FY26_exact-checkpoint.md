# **Task 3: Algorithmic Foundations of Passive Radar & SoO Positioning**

**Assignee:** Miguel Isarraraz

**Period of Performance:** 2 Months

**Status:** Initialized

## **1. Overview**

### **Motivation**

To transition into high-stakes LEO orbit determination based on Signals of Opportunity (SoO) and terrestrial source tracking, we must master the mathematical core of Passive Radar (PR). This task moves away from hardware based experimentation to focus on the Digital Signal Processing (DSP) pipeline. As a deliverable for this task you will develop a comprehensive Jupyter or iPython Notebook that serves as both a prototype library and a theoretical walkthrough for extracting spatial coordinates of objects from the raw electromagnetic data available from the ambient environment

### **Focus: Synthetic Simulation & Algorithmic Proficiency**

You are not required to build a realistic SoO signal from scratch (e.g., modeling specific RDS bit-streams). Instead, you will create a representative synthetic environment. This simulation will provide the ground truth needed to validate your SoO DSP algorithms—specifically time-difference-of-arrival (TDOA) and Doppler shift analysis—before we introduce the noise and non-idealities of real-world captures.

## **2. Task Description**

The final deliverable will be a walkthrough interactive document (e.g. iPython Notebook) organized into the following stages:

### **Stage 1: SoO Synthetic Signal Generator**

Rather than sophisticated modulation, you will model signals as complex-valued Gaussian noise sequences or Linear Frequency Modulated (LFM) chirps. This represents the wideband nature of SoOs without having to dive too deep into coding.

-   **Orig Signal (**s_orig​**):** Generate a random complex sequence to represent the "Direct Path" from a transmitter.

-   **Reflected Signal (**s_ref**):** Create a version of s_orig​ that is:

    1.  **Delayed:** Shifted by N samples to simulate bi-static range.

    2.  **Frequency Shifted:** Multiplied by a complex exponential ej2πfd​t to simulate Doppler shift.

    3.  **Attenuated:** Scaled to represent path loss and the "Direct Path Interference" (DPI) challenge.

-   Objective: Build a function generate_sim_data(delay, doppler_shift, snr) that outputs these two arrays.

### **Stage 2: Signal Acquisition & Matched Filtering**

Implement the algorithms used to detect and track SoO:

-   **Cross-Ambiguity Function (CAF):** Implement the 2D correlation that searches through time-delay (*τ*) and Doppler frequency (*f*<sub>*d*</sub>).

    -   **Implementation Note:** Use Fast Fourier Transforms (FFTs) to optimize the correlation. The CAF is effectively a series of FFTs on the product of the reference and the surveillance signal.

-   **Validation:** Use your Stage 1 simulator to create a heatmap plot. The peak of the heatmap should correspond exactly to the delay and Doppler you injected into the synthetic

### **Stage 3: Bi-static & Multi-static Geometry**

Convert signal "timing" into "location."

-   **The Bi-static Triangle:** Implement functions to convert time-delay into a **Bi-static Range** (*R* = *R*<sub>*t**a**r**g**e**t* − *t**o* − *r**x*</sub> + *R*<sub>*t**a**r**g**e**t* − *t**o* − *t**x*</sub> − *R*<sub>*b**a**s**e**l**i**n**e*</sub>​).

-   **Iso-Range Ellipses:** Write a script to plot the resulting isorange ellipses for a given transmitter/receiver pair.

-   **Multi-static Intersection:** Simulate a target being illuminated by **three** different FM towers. Implement a "Least Squares" solver to find the (*x*, *y*) coordinate where these three ellipses intersect.

### **Stage 4: Analysis of Error (GDOP)**

-   Analyze how the geometry of the towers affects your accuracy.

-   **Experiment:** Move the simulated target closer to the "baseline" (the line between the tower and receiver) and observe how the intersection of the ellipses becomes "blurred." This builds intuition for **Geometric Dilution of Precision (GDOP)**.

## **3. Deliverables**

-   **Walkthrough Notebook:** A single .ipynb file containing:

    1.  The synthetic data generation code.

    2.  Interactive CAF heatmap visualizations.

    3.  A geometric solver that estimates target position from simulated range measurements.

-   **Algorithmic Intuition Whitepaper:** A brief summary (within the notebook) explaining the derivation of the CAF and the impact of GDOP on tracking accuracy.

-   **Synthetic Benchmark Report:** A table showing how accurately the solver recovers () coordinates across various SNR levels (e.g., 10dB, 0dB, -10dB).

## **4. References & Preparation for Simulation**

### **Mathematical Foundations**

-   **Pulse Compression & Correlation:** [<u>PySDR - Pulse Compression</u>](https://www.google.com/search?q=https://pysdr.org/content/pulse_compression.html). This is the best starting point for the Matched Filter logic.

-   **Ambiguity Functions:** Read the "Ambiguity Function" section in *Introduction to Radar Systems* (Skolnik) or [<u>this</u>](https://engineering.purdue.edu/~mrb/resources/AltLectureF/Session_16.pdf) [<u>tutorial</u>](https://engineering.purdue.edu/~mrb/resources/AltLectureF/Session_16.pdf).

### **Guidance for Stage 1 (Simulation)**

-   **Representing Signals:** Use numpy.random.standard_normal for the complex noise. To apply a delay, use numpy.roll or simple array slicing. To apply Doppler, create a time vector t and multiply by np.exp(1j \* 2 \* np.pi \* fd \* t).

-   **Representative vs Realistic:** Do not worry about FM subcarriers or Iridium frame headers. At this stage, a "Signal of Opportunity" is simply any wideband energy source that you can correlate against.

### **Passive Radar Intuition**

-   **Video Lecture:** [<u>Dr. Chris Baker - Passive Radar Principles</u>](https://www.google.com/search?q=https://www.youtube.com/results%3Fsearch_query%3Dchris%2Bbaker%2Bpassive%2Bradar). Watch specifically for the "Cross Ambiguity Function" and "Bi-static Geometry" segments.

-   **LEO Context:** [<u>Satellite Doppler S-Curve Analysis</u>](https://www.rtl-sdr.com/receiving-starlink-beacons-with-an-rtl-sdr-and-lnb/). This helps bridge the gap from stationary towers to moving space-borne sources.
