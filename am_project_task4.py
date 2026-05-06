import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample
import os

# --- Functions ---
def lowpass_filter(signal, cutoff, sample_rate):
    nyq = 0.5 * sample_rate
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(5, normal_cutoff, btype='low')
    return lfilter(b, a, signal)

def get_spectrum(signal, fs):
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    mag = np.abs(np.fft.fft(signal)) / n
    return freqs[:n//2], mag[:n//2]

# --- Setup ---
output_dir = 'task4_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = 'van_wiese-bass-wiggle-297877.wav'
fs_orig, data = wavfile.read(file_path)
if len(data.shape) > 1: data = data[:, 0]
data = data.astype(float) / np.max(np.abs(data))

fs = 100000
num_samples = int(len(data) * fs / fs_orig)
data_resampled = resample(data, num_samples)

message = lowpass_filter(data_resampled, 4000, fs)
t = np.arange(len(message)) / fs
fc = 15000
mu = 0.8
carrier = np.cos(2 * np.pi * fc * t)

# Signals
dsb_lc = (1 + mu * message) * carrier
dsb_sc = message * carrier

# --- STEP 1: Multiplication (Mixing) ---
coherent_carrier = 2 * np.cos(2 * np.pi * fc * t)
mixed_lc = dsb_lc * coherent_carrier
mixed_sc = dsb_sc * coherent_carrier

# Detail: Plot Spectrum After Multiplication (Before Filtering)
f_m, mag_m = get_spectrum(mixed_sc, fs)
plt.figure(figsize=(10, 6))
plt.plot(f_m / 1000, mag_m)
plt.title("Spectrum After Multiplication (Before Lowpass Filter)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.annotate("Recovered Message\nat Baseband", xy=(0, 0.2), xytext=(5, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate("High Frequency Component\nat 2*fc (30 kHz)", xy=(30, 0.1), xytext=(35, 0.2), arrowprops=dict(facecolor='black', shrink=0.05))
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'spectrum_after_mixing.png'))
plt.close()

# --- STEP 2: Filtering ---
recovered_lc_raw = lowpass_filter(mixed_lc, 4000, fs)
recovered_sc_raw = lowpass_filter(mixed_sc, 4000, fs)

# STEP 3: DC Removal
recovered_lc = recovered_lc_raw - np.mean(recovered_lc_raw)
recovered_sc = recovered_sc_raw # SC has no DC

# --- Detailed Visualization (Individual Files) ---
zoom = int(0.02 * fs)
t_ms = t[:zoom] * 1000

# Plot DSB-LC Result
plt.figure(figsize=(10, 6))
plt.plot(t_ms, message[:zoom], label="Original Message", alpha=0.6)
plt.plot(t_ms, recovered_lc[:zoom]/np.max(np.abs(recovered_lc)), '--', label="Recovered DSB-LC")
plt.title("Coherent Detection Detailed Result: DSB-LC")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'detailed_result_lc.png'))
plt.close()

# Plot DSB-SC Result
plt.figure(figsize=(10, 6))
plt.plot(t_ms, message[:zoom], label="Original Message", alpha=0.6)
plt.plot(t_ms, recovered_sc[:zoom]/np.max(np.abs(recovered_sc)), '--', label="Recovered DSB-SC")
plt.title("Coherent Detection Detailed Result: DSB-SC")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'detailed_result_sc.png'))
plt.close()

# Comparison Plot (Coherent vs Envelope)
rectified_lc = np.abs(dsb_lc)
env_demod = lowpass_filter(rectified_lc, 15000, fs)
env_demod = env_demod - np.mean(env_demod)

plt.figure(figsize=(12, 6))
plt.plot(t_ms, message[:zoom], label="Original", alpha=0.3, color='black')
plt.plot(t_ms, recovered_lc[:zoom]/np.max(np.abs(recovered_lc)), label="Coherent Detector", linewidth=2)
plt.plot(t_ms, env_demod[:zoom]/np.max(np.abs(env_demod)), 'r--', label="Envelope Detector")
plt.title("Detailed Comparison: Coherent vs. Envelope Detection Quality")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'detector_quality_comparison.png'))
plt.close()

# --- Advanced Discussion Answers ---
answers = """
TASK 4 ADVANCED DISCUSSION

Question (a): Coherent detection works perfectly for both AM types. It recovers the 
original message signal without the 180-degree phase issues seen in envelope detectors.

Question (b): Quality Comparison
The Coherent Detector provides superior quality. While the envelope detector's quality 
depends heavily on the RC time constant (cutoff frequency), the coherent detector 
isolates the baseband message mathematically, leading to much lower distortion.

Question (c): Advantages
1. Works for DSB-SC (saving transmitter power).
2. Works for overmodulated signals (mu > 1).
3. Better SNR performance in noisy conditions.

Question (d): Implementation Challenge (Synchronization)
The receiver must generate a local carrier cos(wc*t + phi). If phi is not zero, the 
output amplitude is reduced by cos(phi). If the frequency is slightly off, the 
signal will "beat" and fade. 
Solution: To fix this, engineers use a Phase-Locked Loop (PLL) or a Costas Loop 
to "lock" onto the incoming carrier frequency and phase automatically.
"""
with open(os.path.join(output_dir, 'task4_detailed_answers.txt'), 'w') as f:
    f.write(answers)

print(f"Task 4 (Detailed) finished. All results saved in '{output_dir}'.")
