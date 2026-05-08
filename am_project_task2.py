import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# --- Settings ---
fs = 100000       # 100 kHz sampling
fc = 10000       # 10 kHz carrier
duration = 0.1   # 100ms
t = np.arange(int(duration * fs)) / fs

output_dir = 'task2_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Create Multitone Message Signal (m(t))
# Message is the sum of two sinusoids: m(t) = cos(2pi*f1*t) + cos(2pi*f2*t)
f1, f2 = 500, 1500
message_raw = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
# Normalize message so max amplitude is exactly 1.0
message = message_raw / np.max(np.abs(message_raw))

# 2. Modulation and Power Analysis
mu_values = [0.3, 0.5, 0.8, 1.0, 1.2]
efficiencies = []
pm_values = [] # Message Power (Pm)
modulated_signals = {}

# CARRIER-INDEPENDENT FORMULA:
# eta = (mu^2 * Pm_norm) / (1 + mu^2 * Pm_norm)
# Where Pm_norm is the average power of the normalized message m(t)
Pm_norm = np.mean(message**2)
carrier = np.cos(2 * np.pi * fc * t)

print("--- Task 2: Detailed Power Efficiency Analysis ---")
for mu in mu_values:
    # Generate modulated signal: s(t) = (1 + mu*m(t)) * cos(2pi*fc*t)
    s_t = (1 + mu * message) * carrier
    modulated_signals[mu] = s_t
    
    # Message Power Pm = mu^2 * Pm_norm (relative to carrier power)
    Pm = mu**2 * Pm_norm
    # Efficiency eta
    eta = (Pm / (1 + Pm)) * 100
    
    efficiencies.append(eta)
    pm_values.append(Pm)
    
    print(f"mu = {mu:<5} | Pm = {Pm:.4f} | Efficiency = {eta:.2f}%")

# 3. Create Detailed Results Table
table_content = "\n--- Task 2: Detailed Modulation Results ---\n"
table_content += f"{'Modulation Index (mu)':<25} | {'Message Power (Pm)':<20} | {'Efficiency (eta)':<15}\n"
table_content += "-" * 65 + "\n"
for i in range(len(mu_values)):
    table_content += f"{mu_values[i]:<25} | {pm_values[i]:<20.4f} | {efficiencies[i]:<15.2f}%\n"

print(table_content)
with open(os.path.join(output_dir, 'detailed_efficiency_table.txt'), 'w') as f:
    f.write(table_content)

# 4. Detailed Plotting (Question a and b)
# Question (a): Efficiency Plot
plt.figure(figsize=(10, 6))
plt.plot(mu_values, efficiencies, 'o-', color='#1f77b4', linewidth=2.4, markersize=8)
for mu, eta in zip(mu_values, efficiencies):
    plt.annotate(f'{eta:.1f}%', (mu, eta), textcoords='offset points', xytext=(0, 8), ha='center')
plt.title("Modulation Index vs. Power Efficiency", fontsize=14)
plt.xlabel("Modulation Index (mu)", fontsize=12)
plt.ylabel("Power Efficiency (%)", fontsize=12)
plt.ylim(0, max(efficiencies) * 1.18)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'efficiency_vs_mu.png'))
plt.close()

# Question (b): Multi-Waveform Comparison
plt.figure(figsize=(13, 10))
test_mu = [0.5, 1.0, 1.2]
colors = ['#1f77b4', '#2ca02c', '#d62728']
for i, mu in enumerate(test_mu):
    plt.subplot(3, 1, i+1)
    plt.plot(t * 1000, modulated_signals[mu], label=f'Modulated signal', color=colors[i], linewidth=1.5)
    plt.plot(t * 1000, (1 + mu * message), 'k--', alpha=0.85, label='Envelope')
    plt.plot(t * 1000, -(1 + mu * message), 'k--', alpha=0.45)
    plt.title(fr"Time Domain: $\mu = {mu}$" + (" (overmodulation)" if mu > 1 else ""))
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel("Time (ms)")
plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overmodulation_comparison.png'))
plt.close()

print(f"\nTask 2 finished. All detailed results saved in '{output_dir}'.")
