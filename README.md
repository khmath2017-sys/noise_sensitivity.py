
# -*- coding: utf-8 -*-
"""Generate all three figures, zip them, and download the zip file"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import zipfile
import os
import time

# Remove old files if they exist
for f in ['algebraic_plot_new.pdf', 'quintic_nls_H_new.pdf', 'noise_sensitivity_new.pdf', 'figures.zip']:
    if os.path.exists(f):
        os.remove(f)

print("=" * 60)
print("Generating Figure 1: Algebraic blow-up model - H(t)")
print("=" * 60)

# ========== Figure 1 ==========
T0 = 1.0
Tc = 1.0 / T0
dt = 1e-5
t_max = 0.99 * Tc
t = np.arange(0, t_max, dt)
T_exact = 1.0 / (Tc - t)

ell = 0.1
sigma = ell / 2.0

def causal_stepanov_norm(u, t_arr, idx, ell, sigma, dt):
    t_curr = t_arr[idx]
    start_idx = max(0, idx - int(ell / dt))
    t_win = t_arr[start_idx:idx+1]
    weights = np.exp(-(t_curr - t_win)**2 / sigma**2)
    integrand = weights * (u[start_idx:idx+1])**2
    integral = np.trapz(integrand, dx=dt)
    return np.sqrt(integral)

S = np.array([causal_stepanov_norm(T_exact, t, i, ell, sigma, dt) for i in range(len(t))])
S_smooth = savgol_filter(S, window_length=11, polyorder=3, mode='interp')
dS = np.gradient(S_smooth, dt)
H = ell * dS / (S_smooth**2 + 1e-12)

plt.figure(figsize=(8,5))
plt.plot(t, H, 'r-', linewidth=2.5, label=r'$H(t)$')
plt.axhline(y=5, color='gold', linestyle='--', linewidth=2, label='Threshold')
plt.axvline(x=0.9*Tc, color='gray', linestyle=':', linewidth=2, label='90% of Tc')
plt.fill_between(t, 0, H, where=(H>=5), color='red', alpha=0.2, label='Alert region')
plt.xlabel('Time t', fontsize=12)
plt.ylabel(r'$H(t)$', fontsize=12)
plt.title('Algebraic blow-up model: Early warning signal', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('algebraic_plot_new.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: algebraic_plot_new.pdf")

# ========== Figure 2 ==========
print("\n" + "=" * 60)
print("Generating Figure 2: 1D Quintic NLS - S(t) and H(t)")
print("=" * 60)

Tc2 = 1.0
A = 1.0
t2 = np.linspace(0, 0.99*Tc2, 500)
S2 = A / np.sqrt(Tc2 - t2)
dS2 = A / (2 * (Tc2 - t2)**1.5)
ell2 = 1.0
H2 = ell2 * dS2 / (S2**2 + 1e-12)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
ax1.plot(t2, S2, 'b-', linewidth=2.5)
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel(r'$S(t)$', fontsize=12)
ax1.set_title('Sobolev-Stepanov norm evolution', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(t2, H2, 'r-', linewidth=2.5, label=r'$H(t)$')
ax2.axhline(y=5, color='gold', linestyle='--', linewidth=2, label='Threshold')
ax2.fill_between(t2, 0, H2, where=(H2>=5), color='red', alpha=0.2, label='Early warning zone')
ax2.set_xlabel('Time t', fontsize=12)
ax2.set_ylabel(r'$H(t)$', fontsize=12)
ax2.set_title('Early warning signal H(t) - 1D Quintic NLS', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('quintic_nls_H_new.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: quintic_nls_H_new.pdf")

# ========== Figure 3 ==========
print("\n" + "=" * 60)
print("Generating Figure 3: Noise sensitivity comparison (1000 realizations)")
print("=" * 60)

np.random.seed(2026)

def compute_relative_changes_with_std():
    noise_levels = [0, 5, 10]
    mean_H1, std_H1 = [], []
    mean_L2, std_L2 = [], []
    mean_Sw, std_Sw = [], []
    for noise in noise_levels:
        H1_changes, L2_changes, Sw_changes = [], [], []
        for _ in range(1000):
            x = np.linspace(0, 1, 200)
            clean = np.exp(-((x-0.5)/0.05)**2)
            if noise > 0:
                noisy = clean + (noise/100.0) * np.random.normal(0, np.std(clean), len(clean))
            else:
                noisy = clean
            dx = x[1] - x[0]
            L2_clean = np.sqrt(np.sum(clean**2) * dx)
            L2_noisy = np.sqrt(np.sum(noisy**2) * dx)
            L2_changes.append(abs(L2_noisy - L2_clean) / L2_clean)
            grad_clean = np.gradient(clean, dx)
            grad_noisy = np.gradient(noisy, dx)
            H1_clean = np.sqrt(L2_clean**2 + np.sum(grad_clean**2) * dx)
            H1_noisy = np.sqrt(L2_noisy**2 + np.sum(grad_noisy**2) * dx)
            H1_changes.append(abs(H1_noisy - H1_clean) / H1_clean)
            window_len = 20
            center = 100
            start = max(0, center - window_len // 2)
            end = min(len(x), center + window_len // 2)
            weights = np.exp(-((x[start:end] - 0.5)**2) / (0.05**2))
            Sw_clean = np.sqrt(np.sum(weights * clean[start:end]**2) * dx)
            Sw_noisy = np.sqrt(np.sum(weights * noisy[start:end]**2) * dx)
            Sw_changes.append(abs(Sw_noisy - Sw_clean) / Sw_clean)
        mean_L2.append(np.mean(L2_changes) * 100)
        std_L2.append(np.std(L2_changes) * 100)
        mean_H1.append(np.mean(H1_changes) * 100)
        std_H1.append(np.std(H1_changes) * 100)
        mean_Sw.append(np.mean(Sw_changes) * 100)
        std_Sw.append(np.std(Sw_changes) * 100)
    return noise_levels, mean_L2, std_L2, mean_H1, std_H1, mean_Sw, std_Sw

noise_lev, mL2, sL2, mH1, sH1, mSw, sSw = compute_relative_changes_with_std()

x_vals = np.arange(len(noise_lev))
width = 0.25
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x_vals - width, mH1, width, yerr=sH1, capsize=5, label='H^1',
       color='salmon', error_kw={'ecolor': 'darkred', 'linewidth': 1.5})
ax.bar(x_vals, mL2, width, yerr=sL2, capsize=5, label='L^2',
       color='skyblue', error_kw={'ecolor': 'darkblue', 'linewidth': 1.5})
ax.bar(x_vals + width, mSw, width, yerr=sSw, capsize=5, label='S_w^2 (ours)',
       color='lightgreen', error_kw={'ecolor': 'darkgreen', 'linewidth': 1.5})
ax.set_xlabel('Noise level (%)', fontsize=12)
ax.set_ylabel('Relative norm change (%)', fontsize=12)
ax.set_title('Norm sensitivity comparison (1000 realizations, error bars = std)', fontsize=14)
ax.set_xticks(x_vals)
ax.set_xticklabels(noise_lev)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('noise_sensitivity_new.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: noise_sensitivity_new.pdf")

# ========== Create ZIP file and download ==========
print("\n" + "=" * 60)
print("Creating ZIP archive and downloading...")
print("=" * 60)

with zipfile.ZipFile('figures.zip', 'w') as zipf:
    zipf.write('algebraic_plot_new.pdf')
    zipf.write('quintic_nls_H_new.pdf')
    zipf.write('noise_sensitivity_new.pdf')

from google.colab import files
files.download('figures.zip')

print("\n✅ All three figures have been zipped and downloaded as 'figures.zip'")
print("Extract the ZIP file to obtain the three high-resolution PDF figures.")
