import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 140,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})

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


def find_representative_window(signal, sample_rate, window_seconds=0.02):
    window_samples = max(1, int(window_seconds * sample_rate))
    energy = np.convolve(signal ** 2, np.ones(window_samples), mode='valid')
    start = int(np.argmax(energy))
    end = min(len(signal), start + window_samples)
    return start, end


def normalize_for_display(signal):
    max_value = np.max(np.abs(signal))
    if max_value == 0:
        return signal
    return signal / max_value


def save_wav_file(file_name, signal, sample_rate):
    clipped = np.clip(signal, -1.0, 1.0)
    int16_signal = np.int16(clipped * 32767)
    wavfile.write(os.path.join(output_dir, file_name), sample_rate, int16_signal)

# --- Setup ---
output_dir = 'task4_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = 'van_wiese-bass-wiggle-297877.wav'
fs_orig, data = wavfile.read(file_path)
if len(data.shape) > 1: data = data[:, 0]
data = data.astype(float) / np.max(np.abs(data))
target_samples = len(data)

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

# Choose a representative window so the time-domain plots show the message shape clearly.
start_sample, end_sample = find_representative_window(message, fs, window_seconds=0.02)
t_window_ms = (t[start_sample:end_sample] - t[start_sample]) * 1000

# Save reference audio for comparison and listening.
message_audio = data

# --- STEP 1: Multiplication (Mixing) ---
coherent_carrier = 2 * np.cos(2 * np.pi * fc * t)
mixed_lc = dsb_lc * coherent_carrier
mixed_sc = dsb_sc * coherent_carrier

# Detail: Plot Spectrum After Multiplication (Before Filtering)
f_m, mag_m = get_spectrum(mixed_sc, fs)
plt.figure(figsize=(10, 6))
plt.plot(f_m / 1000, mag_m, color='#1f77b4', linewidth=1.4)
plt.title("Spectrum After Multiplication (Before Lowpass Filter)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.annotate("Recovered Message\nat Baseband", xy=(0, 0.2), xytext=(5, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate("High Frequency Component\nat 2*fc (30 kHz)", xy=(30, 0.1), xytext=(35, 0.2), arrowprops=dict(facecolor='black', shrink=0.05))
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spectrum_after_mixing.png'))
plt.close()

# --- STEP 2: Filtering ---
recovered_lc_raw = lowpass_filter(mixed_lc, 4000, fs)
recovered_sc_raw = lowpass_filter(mixed_sc, 4000, fs)

# STEP 3: DC Removal
recovered_lc = recovered_lc_raw - np.mean(recovered_lc_raw)
recovered_sc = recovered_sc_raw # SC has no DC

# Scale recovered signals for audio export and save the demodulated wav files.
recovered_lc_audio = resample(recovered_lc, target_samples)
recovered_sc_audio = resample(recovered_sc, target_samples)
save_wav_file('demodulated_dsb_lc_coherent.wav', normalize_for_display(recovered_lc_audio), fs_orig)
save_wav_file('demodulated_dsb_sc_coherent.wav', normalize_for_display(recovered_sc_audio), fs_orig)
save_wav_file('original_message_reference.wav', normalize_for_display(message_audio), fs_orig)

# --- Detailed Visualization (Individual Files) ---
zoom = end_sample - start_sample

# Plot DSB-LC Result
plt.figure(figsize=(12, 6))
plt.plot(t_window_ms, normalize_for_display(message[start_sample:end_sample]), label="Original Message", linewidth=2.2, color='#1f77b4')
plt.plot(t_window_ms, normalize_for_display(recovered_lc[start_sample:end_sample]), '--', label="Recovered DSB-LC", linewidth=2.0, color='#ff7f0e')
plt.title("Task 4 (a): Coherent Detection on DSB-LC")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'detailed_result_lc.png'))
plt.close()

# Plot DSB-SC Result
plt.figure(figsize=(12, 6))
plt.plot(t_window_ms, normalize_for_display(message[start_sample:end_sample]), label="Original Message", linewidth=2.2, color='#1f77b4')
plt.plot(t_window_ms, normalize_for_display(recovered_sc[start_sample:end_sample]), '--', label="Recovered DSB-SC", linewidth=2.0, color='#2ca02c')
plt.title("Task 4 (a): Coherent Detection on DSB-SC")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'detailed_result_sc.png'))
plt.close()

# Comparison Plot (Coherent vs Envelope)
rectified_lc = np.abs(dsb_lc)
env_demod = lowpass_filter(rectified_lc, 15000, fs)
env_demod = env_demod - np.mean(env_demod)

message_window = normalize_for_display(message[start_sample:end_sample])
coherent_window = normalize_for_display(recovered_lc[start_sample:end_sample])
envelope_window = normalize_for_display(env_demod[start_sample:end_sample])

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t_window_ms, message_window, label="Original", linewidth=2.2, color='black', alpha=0.75)
axes[0].plot(t_window_ms, coherent_window, label="Coherent Demodulation", linewidth=2.3, color='#1f77b4')
axes[0].set_title("Task 4 (b): Coherent Demodulation vs Original (DSB-LC)")
axes[0].set_ylabel("Normalized amplitude")
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.35)

axes[1].plot(t_window_ms, message_window, label="Original", linewidth=2.2, color='black', alpha=0.75)
axes[1].plot(t_window_ms, envelope_window, 'r--', label="Envelope Demodulation", linewidth=2.0)
axes[1].set_title("Task 4 (b): Envelope Demodulation vs Original (DSB-LC)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Normalized amplitude")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.35)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'detector_quality_comparison.png'))
fig.savefig(os.path.join(output_dir, 'detector_comparison.png'))
plt.close(fig)

# Compact summary plot for the assignment question.
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_window_ms, normalize_for_display(message[start_sample:end_sample]), label="Original", linewidth=2.0, color='black')
plt.plot(t_window_ms, normalize_for_display(recovered_lc[start_sample:end_sample]), '--', label="Recovered DSB-LC", linewidth=2.0, color='#ff7f0e')
plt.plot(t_window_ms, normalize_for_display(recovered_sc[start_sample:end_sample]), ':', label="Recovered DSB-SC", linewidth=2.2, color='#2ca02c')
plt.title("Task 4 (a): Coherent Detection Output")
plt.ylabel("Normalized amplitude")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.35)

plt.subplot(2, 1, 2)
plt.plot(t_window_ms, normalize_for_display(message[start_sample:end_sample]), label="Original", linewidth=2.0, color='black', alpha=0.7)
plt.plot(t_window_ms, normalize_for_display(recovered_lc[start_sample:end_sample]), label="Coherent", linewidth=2.2, color='#1f77b4')
plt.plot(t_window_ms, normalize_for_display(env_demod[start_sample:end_sample]), 'r--', label="Envelope", linewidth=2.0)
plt.title("Task 4 (b): Quality Comparison for DSB-LC")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'coherent_results.png'))
plt.close()

print(f"Task 4 (Detailed) finished. All results saved in '{output_dir}'.")
