import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

# Setup
file_path = 'van_wiese-bass-wiggle-297877.wav'
output_dir = 'task1_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 140,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})

# Helper Functions 

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

def save_spectrum_comparison(all_signals, sample_rate, center_frequency, output_path, use_db=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()
    color_map = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    for axis, (name, signal), color in zip(axes, all_signals.items(), color_map):
        n = len(signal)
        freqs = np.fft.fftfreq(n, 1 / sample_rate)
        fft_vals = np.fft.fft(signal)

        positive_mask = freqs >= 0
        positive_freqs = freqs[positive_mask]
        positive_magnitude = np.abs(fft_vals[positive_mask])

        if use_db:
            mag_plot = 20 * np.log10(positive_magnitude + 1e-9)
            axis.plot(positive_freqs / 1000, mag_plot, color=color, linewidth=1.2)
            axis.set_ylabel('Magnitude (dB)')
        else:
            axis.plot(positive_freqs / 1000, positive_magnitude, color=color, linewidth=1.2)
            axis.set_ylabel('Magnitude')
            
            # Dynamic Zoom for Linear plots
            sideband_mask = (positive_freqs < center_frequency - 100) | (positive_freqs > center_frequency + 100)
            if np.any(sideband_mask):
                max_sideband = np.max(positive_magnitude[sideband_mask])
                axis.set_ylim(0, max_sideband * 1.6)
                axis.annotate(f"Carrier: {np.max(positive_magnitude):,.0f}", 
                             xy=(center_frequency/1000, max_sideband * 1.3),
                             fontsize=8, ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))

        axis.set_title(name)
        axis.set_xlim(center_frequency / 1000 - 8, center_frequency / 1000 + 8)
        axis.grid(True, linestyle='--', alpha=0.4)
        axis.set_xlabel('Frequency (kHz)')

    fig.suptitle('AM Spectrum Comparison (' + ('dB Scale' if use_db else 'Linear - Zoomed Sidebands') + ')')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)

# Main Code

# Load the input file and then and Normalize it to avoid overmodulation
fs, data = wavfile.read(file_path)
if len(data.shape) > 1:
    data = data[:, 0]  
data = data.astype(float) / np.max(np.abs(data))
duration = len(data) / fs
print(f"File loaded: {duration:.2f} seconds, Sample Rate: {fs} Hz")

# Filtering
filtered_audio = bandlimit_filter(data, 4000, fs)
message_bandwidth_hz = estimate_99_power_bandwidth(filtered_audio, fs)
transmitted_bandwidth_hz = 2 * message_bandwidth_hz

# Modulation Setup
t = np.arange(len(data)) / fs
fc = 15000 
carrier = np.cos(2 * np.pi * fc * t)

# DSB-SC
dsb_sc = filtered_audio * carrier

# DSB-LC with varying mu and Ac = abs(min(data))/mu
mu_values = [0.5, 0.8, 1.0]
all_signals = {'DSB-SC': dsb_sc}

min_val = np.abs(np.min(filtered_audio))
print(f"Minimum message value (absolute): {min_val:.4f}")

for mu in mu_values:
    Ac = min_val / mu
    all_signals[f'DSB-LC (mu={mu})'] = (Ac + filtered_audio) * carrier
    print(f"For mu={mu}, Carrier Amplitude Ac={Ac:.4f}")

# Visualization and Bandwidth Analysis
results_bw = {}
zoom_duration = 0.05
zoom_samples = int(zoom_duration * fs)
start_sample = int(len(data) // 2)
end_sample = start_sample + zoom_samples

for name, signal in all_signals.items():
    # Frequency Domain
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_vals = np.fft.fft(signal)
    
    mask = freqs >= 0
    f_pos = freqs[mask]
    mag_pos = np.abs(fft_vals[mask])
    
    bw = transmitted_bandwidth_hz
    results_bw[name] = bw
    
    # Combined Spectrum Plot (Linear and dB)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Linear Plot (Zoomed)
    ax1.plot(f_pos / 1000, mag_pos, color='#1f77b4', linewidth=1.6)
    ax1.set_title(f"Frequency Spectrum (Linear): {name}\n99% Power BW: {bw/1000:.2f} kHz")
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Magnitude")
    ax1.set_xlim(fc/1000 - 8, fc/1000 + 8)
    
    # Dynamic Zoom: Find max magnitude in sidebands to set Y-limit
    # Mask out the carrier spike (fc +/- 100 Hz)
    sideband_mask = (f_pos < fc - 100) | (f_pos > fc + 100)
    if np.any(sideband_mask):
        max_sideband = np.max(mag_pos[sideband_mask])
        ax1.set_ylim(0, max_sideband * 1.5)
        ax1.annotate(f"Carrier spike truncated\n(Full height: {np.max(mag_pos):,.0f})", 
                     xy=(fc/1000, max_sideband * 1.2), 
                     xytext=(fc/1000 + 2, max_sideband * 1.4),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax1.grid(True, linestyle='--', alpha=0.4)

    # dB Plot
    mag_db = 20 * np.log10(mag_pos + 1e-9)
    ax2.plot(f_pos / 1000, mag_db, color='#d62728', linewidth=1.4)
    ax2.set_title(f"Frequency Spectrum (dB): {name}")
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_xlim(fc/1000 - 10, fc/1000 + 10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace('=', '_').replace(' ', '_')}_Spectrum_Combined.png"))
    plt.close()

    # Time Domain Plot
    plt.figure(figsize=(10, 6))
    t_zoom = t[start_sample:end_sample] * 1000 
    plt.plot(t_zoom, signal[start_sample:end_sample], color='#ff7f0e', linewidth=1.6)
    plt.title(f"Time Domain Waveform (50ms): {name}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace('=', '_').replace(' ', '_')}_TimeDomain.png"))
    plt.close()

# Generate Final Comparison Plots
# Linear Scale (Zoomed on sidebands)
save_spectrum_comparison(all_signals, fs, fc, os.path.join(output_dir, 'spectrum_comparison.png'), use_db=False)

# dB Scale
save_spectrum_comparison(all_signals, fs, fc, os.path.join(output_dir, 'spectrum_comparison_db.png'), use_db=True)

# Analysis Table
results_data = []
for name, signal in all_signals.items():
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    mag = np.abs(fft_vals) / n
    total_power = np.sum(mag**2)
    
    freqs = np.fft.fftfreq(n, 1/fs)
    fc_idx = np.argmin(np.abs(np.abs(freqs) - fc))
    carrier_power = np.sum(mag[fc_idx-2:fc_idx+3]**2) if 'LC' in name else 0
    sideband_power = total_power - carrier_power
    efficiency = (sideband_power / total_power) * 100
    
    results_data.append({
        'name': name,
        'bw': transmitted_bandwidth_hz / 1000,
        'efficiency': efficiency,
        'total_p': total_power
    })

header = f"{'Modulation Type':<25} | {'99% BW (kHz)':<15} | {'Efficiency (%)':<15}\n"
separator = "-" * 60 + "\n"
table_content = "\n--- Detailed Modulation Analysis ---\n" + header + separator
for item in results_data:
    table_content += f"{item['name']:<25} | {item['bw']:<15.2f} | {item['efficiency']:<15.2f}\n"

print(table_content)
with open(os.path.join(output_dir, 'detailed_results.txt'), 'w') as f:
    f.write(table_content)

# Bandwidth Comparison Table
bw_table = "\n--- Bandwidth Comparison Table ---\n"
bw_table += f"{'Modulation Type':<25} | {'99% BW (kHz)':<15}\n"
bw_table += "-" * 45 + "\n"
for item in results_data:
    bw_table += f"{item['name']:<25} | {item['bw']:<15.2f}\n"

with open(os.path.join(output_dir, 'bandwidth_results.txt'), 'w') as f:
    f.write(bw_table)

print(f"\nTask 1 finished. Results saved in '{output_dir}'.")
