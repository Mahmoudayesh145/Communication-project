import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
file_path = 'van_wiese-bass-wiggle-297877.wav'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 140,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})

# -------------------------------Functions------------------------------

# Transmitter: Bandlimit Filter
# AM radio typically uses a 4kHz - 5kHz bandwidth for voice/audio
def bandlimit_filter(signal, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)


def estimate_99_power_bandwidth(signal, sample_rate):
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1 / sample_rate)
    fft_vals = np.fft.fft(signal)

    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_power = np.abs(fft_vals[positive_mask]) ** 2

    cumulative_power = np.cumsum(positive_power)
    total_power = cumulative_power[-1]

    lower_index = np.searchsorted(cumulative_power, 0.005 * total_power)
    upper_index = np.searchsorted(cumulative_power, 0.995 * total_power)
    return positive_freqs[upper_index] - positive_freqs[lower_index]


def save_spectrum_comparison(all_signals, sample_rate, center_frequency, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    color_map = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    for axis, (name, signal), color in zip(axes, all_signals.items(), color_map):
        n = len(signal)
        freqs = np.fft.fftfreq(n, 1 / sample_rate)
        fft_vals = np.fft.fft(signal)

        positive_mask = freqs >= 0
        positive_freqs = freqs[positive_mask]
        positive_magnitude = np.abs(fft_vals[positive_mask])

        axis.plot(positive_freqs / 1000, positive_magnitude, color=color, linewidth=1.4)
        axis.set_title(name)
        axis.set_xlim(center_frequency / 1000 - 8, center_frequency / 1000 + 8)
        axis.grid(True, linestyle='--', alpha=0.4)
        axis.set_xlabel('Frequency (kHz)')
        axis.set_ylabel('Magnitude')

    fig.suptitle('AM Spectrum Comparison')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)


# -------------------------------------------------------------

# Load the Audio File
fs, data = wavfile.read(file_path)

# Ensure it is Mono as it must has only one column 
if len(data.shape) > 1:
    data = data[:, 0]  # Take one channel if not mono

# Normalize for processing (range -1 to 1 for the magnitude of the signal) 
# in order to avoid overflow (overmodulation)
data = data.astype(float) / np.max(np.abs(data))

# Calculate Duration ==> samples/samplerate
duration = len(data) / fs
print(f"File loaded: {duration:.2f} seconds, Sample Rate: {fs} Hz")



# Apply filter (e.g., 4000 Hz limit)
filtered_audio = bandlimit_filter(data, 4000, fs)

# Estimate message bandwidth once, then report the transmitted bandwidth for AM
# as twice the occupied message bandwidth. This keeps the comparison consistent
# across DSB-SC and DSB-LC because modulation shifts the spectrum but does not
# change the sideband extent.
message_bandwidth_hz = estimate_99_power_bandwidth(filtered_audio, fs)
transmitted_bandwidth_hz = 2 * message_bandwidth_hz

# Create Time Axis
t = np.arange(len(data)) / fs

# Define Carrier Frequency 
# (Must be higher than message bandwidth, e.g., 15-20kHz for this fs)
fc = 15000 
carrier = np.cos(2 * np.pi * fc * t)

# Modulation: DSB-SC (Double Sideband - Suppressed Carrier)
dsb_sc = filtered_audio * carrier

# 1. Setup for Multi-Index Modulation
mu_values = [0.5, 0.8, 1.0]
all_signals = {'DSB-SC': dsb_sc}

# Store all DSB-LC signals in a dictionary
for m in mu_values:
    all_signals[f'DSB-LC (mu={m})'] = (1 + m * filtered_audio) * carrier

import os

# Create directory if it doesn't exist
output_dir = 'task1_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 2. Visualization and Bandwidth Analysis
results_bw = {}

# Time window for Zoom (50ms)
zoom_duration = 0.05
zoom_samples = int(zoom_duration * fs)
start_sample = int(len(data) // 2) # Take from the middle
end_sample = start_sample + zoom_samples

for name, signal in all_signals.items():
    # --- A. Frequency Domain (FFT) ---
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_vals = np.fft.fft(signal)
    
    mask = freqs >= 0
    f_pos = freqs[mask]
    mag_pos = np.abs(fft_vals[mask])
    
    # The transmitted AM bandwidth depends on the message bandwidth, not on the
    # presence or absence of the carrier line.
    bw = transmitted_bandwidth_hz
    results_bw[name] = bw
    
    # Save Spectrum Plot
    plt.figure(figsize=(10, 6))
    plt.plot(f_pos / 1000, mag_pos, color='#1f77b4', linewidth=1.6)
    plt.title(f"Frequency Spectrum: {name}\n99% Power Bandwidth: {bw/1000:.2f} kHz")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude")
    plt.xlim(fc/1000 - 8, fc/1000 + 8) # Focused zoom around fc
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace('=', '_')}_Spectrum.png"))
    plt.close()

    # --- B. Time Domain (Zoomed) ---
    plt.figure(figsize=(10, 6))
    t_zoom = t[start_sample:end_sample] * 1000 # Convert to ms
    plt.plot(t_zoom, signal[start_sample:end_sample], color='#ff7f0e', linewidth=1.6)
    plt.title(f"Time Domain Waveform (50ms): {name}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace('=', '_')}_TimeDomain.png"))
    plt.close()

save_spectrum_comparison(all_signals, fs, fc, os.path.join(output_dir, 'spectrum_comparison.png'))

print(f"\nAll detailed plots saved to the '{output_dir}' folder.")

# 3. Print and Save Detailed Comparison Table
results_data = []

for name, signal in all_signals.items():
    # Frequency Analysis
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_vals = np.fft.fft(signal)
    mag = np.abs(fft_vals) / n
    
    # Power Calculation
    total_power = np.sum(mag**2)
    
    # Carrier Power (Find peak at fc)
    fc_idx = np.argmin(np.abs(np.abs(freqs) - fc))
    # Sum a small window around fc to capture the carrier spike
    carrier_power = np.sum(mag[fc_idx-2:fc_idx+3]**2) if 'LC' in name else 0
    sideband_power = total_power - carrier_power
    efficiency = (sideband_power / total_power) * 100
    
    results_data.append({
        'name': name,
        'bw': transmitted_bandwidth_hz / 1000,
        'efficiency': efficiency,
        'total_p': total_power
    })

# Format Detailed Table
header = f"{'Modulation Type':<20} | {'99% BW (kHz)':<15} | {'Efficiency (%)':<15}\n"
separator = "-" * 55 + "\n"
table_content = "\n--- Detailed Modulation Analysis ---\n" + header + separator

for item in results_data:
    table_content += f"{item['name']:<20} | {item['bw']:<15.2f} | {item['efficiency']:<15.2f}\n"

print(table_content)

with open(os.path.join(output_dir, 'detailed_results.txt'), 'w') as f:
    f.write(table_content)
print(f"Detailed table saved to {output_dir}/detailed_results.txt")

# Save the bandwidth comparison table in a separate file.
bandwidth_table = [
    "\n--- Bandwidth Comparison Table ---",
    f"{'Modulation Type':<20} | {'99% BW (kHz)':<15}",
    "----------------------------------------",
]

for item in results_data:
    bandwidth_table.append(f"{item['name']:<20} | {item['bw']:<15.2f}")

with open(os.path.join(output_dir, 'bandwidth_results.txt'), 'w') as file_handle:
    file_handle.write('\n'.join(bandwidth_table))

print(f"Bandwidth table saved to {output_dir}/bandwidth_results.txt")



