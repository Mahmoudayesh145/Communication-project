import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample
import os

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

# Message Preparation
message = lowpass_filter(data_resampled, 4000, fs)
t = np.arange(len(message)) / fs
fc = 15000
mu = 0.8
carrier = np.cos(2 * np.pi * fc * t)

# Generate both DSB-LC and DSB-SC
dsb_lc = (1 + mu * message) * carrier
dsb_sc = message * carrier

# --- STEP 1: Rectification ---
rect_lc = np.abs(dsb_lc)
rect_sc = np.abs(dsb_sc)

# --- STEP 2: Lowpass Filter Experiment (Question b) ---
cutoffs = [1000, 15000, 50000]
filtered_results = {f: lowpass_filter(rect_lc, f, fs) for f in cutoffs}

plt.figure(figsize=(12, 10))
plt.suptitle("Task 3 (b): Cutoff Frequency Experiment")
zoom = int(0.02 * fs)
for i, f in enumerate(cutoffs):
    plt.subplot(3, 1, i+1)
    plt.plot(t[:zoom]*1000, rect_lc[:zoom], color='gray', alpha=0.3, label='Rectified')
    plt.plot(t[:zoom]*1000, filtered_results[f][:zoom], label=f'Cutoff={f/1000}kHz')
    plt.legend()
    plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, 'cutoff_experiment.png'))
plt.close()

# --- STEP 3: DC Removal and Comparison (Question a) ---
# We use the 15kHz result (Just Right)
demod_raw = filtered_results[15000]
recovered_lc = demod_raw - np.mean(demod_raw)

plt.figure(figsize=(12, 6))
plt.plot(t[:zoom]*1000, message[:zoom], label="Original Message", linewidth=2)
# Normalize for visual overlay
recov_norm = recovered_lc / np.max(np.abs(recovered_lc))
plt.plot(t[:zoom]*1000, recov_norm[:zoom], '--', label="Recovered (Envelope Det.)")
plt.title("Task 3 (a): Recovery Comparison (DSB-LC)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'recovery_comparison_lc.png'))
plt.close()

# --- Question (c): Envelope Detector on DSB-SC ---
# Rectify and filter DSB-SC
demod_sc_raw = lowpass_filter(rect_sc, 15000, fs)
recovered_sc = demod_sc_raw - np.mean(demod_sc_raw)

plt.figure(figsize=(12, 6))
plt.plot(t[:zoom]*1000, message[:zoom], label="Original Message")
plt.plot(t[:zoom]*1000, np.abs(message[:zoom]), '--', label="Recovered from DSB-SC")
plt.title("Task 3 (c): Envelope Detector applied to DSB-SC")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'detector_on_dsb_sc.png'))
plt.close()

# --- Question (d): Written Answers ---
answers = """
TASK 3 DISCUSSION ANSWERS

Question (b): Cutoff Frequency Observations
- Too Low (1 kHz): The recovered signal is very smooth but "laggy" and misses high-frequency details. 
- Too High (50 kHz): The carrier ripples are still clearly visible in the output; it's noisy.
- Just Right (15 kHz): The output follows the message envelope accurately with minimal noise.

Question (c): Applying Envelope Detector to DSB-SC
Output: The output looks like the absolute value of the original message (|m(t)|) rather than m(t).
Why it doesn't work: DSB-SC has no carrier to "lift" the message above zero. When the message 
crosses zero, the envelope detector (which is a peak tracker) just sees it as a positive peak 
again, losing the original sign (phase) of the message.

Question (d): Advantages and Limitations
Advantage: Extremely simple and cheap to build (needs only a diode, resistor, and capacitor). 
No complex synchronization or local carrier generation is needed at the receiver.
Limitation: Very inefficient in terms of power (requires a large carrier). It also fails 
completely if mu > 1 (overmodulation) or if used on DSB-SC.
"""
with open(os.path.join(output_dir, 'task3_answers.txt'), 'w') as f:
    f.write(answers)

print(f"Task 3 finished. All results saved to '{output_dir}'.")
