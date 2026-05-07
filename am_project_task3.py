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
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})

# --- Setup ---
output_dir = 'task3_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = 'van_wiese-bass-wiggle-297877.wav'
fs_orig, data = wavfile.read(file_path)
if len(data.shape) > 1: data = data[:, 0]
data = data.astype(float) / np.max(np.abs(data))

# Upsample to 100kHz to allow the "50kHz" cutoff experiment
fs = 100000
num_samples = int(len(data) * fs / fs_orig)
data_resampled = resample(data, num_samples)

def lowpass_filter(signal, cutoff, sample_rate):
    nyq = 0.5 * sample_rate
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(5, normal_cutoff, btype='low')
    return lfilter(b, a, signal)


def find_representative_window(signal, sample_rate, window_seconds=0.004):
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

# Message Preparation
message = lowpass_filter(data_resampled, 4000, fs)
t = np.arange(len(message)) / fs
fc = 15000
mu = 0.8
carrier = np.cos(2 * np.pi * fc * t)

# Generate both DSB-LC and DSB-SC
dsb_lc = (1 + mu * message) * carrier
dsb_sc = message * carrier

# Use a representative active window so the envelope behavior is visible in the plots.
start_sample, end_sample = find_representative_window(message, fs, window_seconds=0.004)
window_t = t[start_sample:end_sample] * 1000

# --- STEP 1: Rectification ---
rect_lc = np.abs(dsb_lc)
rect_sc = np.abs(dsb_sc)

# --- STEP 2: Lowpass Filter Experiment (Question b) ---
cutoffs = [1000, 15000, 50000]
filtered_results = {f: lowpass_filter(rect_lc, f, fs) for f in cutoffs}

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.suptitle("Task 3 (b): Lowpass Filter Cutoff Frequency Experiment", fontweight='bold')
cutoff_notes = {
    1000: "Too low: smooth but loses message detail",
    15000: "Just right: carrier removed, message preserved",
    50000: "Too high: carrier ripple leaks through"
}
cutoff_colors = {1000: '#1f77b4', 15000: '#2ca02c', 50000: '#d62728'}
for i, f in enumerate(cutoffs):
    axes[i].plot(window_t, normalize_for_display(rect_lc[start_sample:end_sample]),
                 color='0.82', linewidth=1.0, label='Rectified input')
    axes[i].plot(window_t, normalize_for_display(filtered_results[f][start_sample:end_sample]),
                 color=cutoff_colors[f], linewidth=2.2,
                 label=f'Lowpass output ({f/1000:.1f} kHz)')
    axes[i].set_ylabel('Norm. amp.')
    axes[i].legend(loc='upper right')
    axes[i].text(0.99, 0.08, cutoff_notes[f], transform=axes[i].transAxes,
                 ha='right', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='0.85'))
    axes[i].grid(True, alpha=0.35)
axes[-1].set_xlabel('Time (ms)')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, 'cutoff_experiment.png'))
plt.close(fig)

# --- STEP 3: DC Removal and Comparison (Question a) ---
# We use the 15kHz result (Just Right)
demod_raw = filtered_results[15000]
recovered_lc = demod_raw - np.mean(demod_raw)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
message_window = normalize_for_display(message[start_sample:end_sample])
recovered_window = normalize_for_display(recovered_lc[start_sample:end_sample])
error_window = message_window - recovered_window
axes[0].plot(window_t, message_window, color='#1f77b4', linewidth=2.2, label='Original message')
axes[0].plot(window_t, recovered_window, color='#ff7f0e', linewidth=2.0, linestyle='--', label='Envelope detector output')
axes[0].set_title('Task 3 (a): DSB-LC Envelope Recovery')
axes[0].set_ylabel('Normalized amplitude')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.35)
axes[1].plot(window_t, error_window, color='#d62728', linewidth=1.8)
axes[1].axhline(0, color='0.3', linewidth=1)
axes[1].set_title('Recovery error (original - output)')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Error')
axes[1].grid(True, alpha=0.35)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'recovery_comparison_lc.png'))
plt.close(fig)

# --- Question (c): Envelope Detector on DSB-SC ---
# Rectify and filter DSB-SC
demod_sc_raw = lowpass_filter(rect_sc, 15000, fs)
recovered_sc = demod_sc_raw - np.mean(demod_sc_raw)

# The envelope detector on DSB-SC fails because it recovers |m(t)| 
# instead of m(t), losing the phase information
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
message_sc_window = normalize_for_display(message[start_sample:end_sample])
recovered_sc_window = normalize_for_display(recovered_sc[start_sample:end_sample])
abs_message_window = normalize_for_display(np.abs(message[start_sample:end_sample]))
axes[0].plot(window_t, message_sc_window, label='Original message', linewidth=2.2, color='#2ca02c')
axes[0].plot(window_t, recovered_sc_window, '--', label='Envelope detector output', linewidth=2.0, color='#d62728')
axes[0].set_title('Task 3 (c): Envelope Detector Applied to DSB-SC')
axes[0].set_ylabel('Normalized amplitude')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.35)
axes[0].text(0.02, 0.08, 'Output loses the sign information', transform=axes[0].transAxes,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='0.85'))

axes[1].plot(window_t, message_sc_window, label='Original message m(t)', linewidth=2.2, color='#2ca02c')
axes[1].plot(window_t, abs_message_window, '--', label='Magnitude |m(t)|', linewidth=2.0, color='#ff7f0e')
axes[1].set_title('Why it fails: detector tracks magnitude, not sign')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Normalized amplitude')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.35)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'detector_on_dsb_sc.png'))
plt.close(fig)

print(f"Task 3 finished. All results saved to '{output_dir}'.")
