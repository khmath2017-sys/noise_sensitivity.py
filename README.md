import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2026)

def compute_relative_changes_with_std():
    noise_levels = [0, 5, 10]
    mean_H1, std_H1, mean_L2, std_L2, mean_Sw, std_Sw = [], [], [], [], [], []

    for noise in noise_levels:
        H1_changes, L2_changes, Sw_changes = [], [], []
        for _ in range(1000):
            x = np.linspace(0, 1, 200)
            clean = np.exp(-((x-0.5)/0.05)**2)
            if noise > 0:
                noisy = clean + (noise/100.0) * np.random.normal(0, np.std(clean), len(clean))
            else:
                noisy = clean
            dx = x[1]-x[0]
            L2_clean = np.sqrt(np.sum(clean**2)*dx)
            L2_noisy = np.sqrt(np.sum(noisy**2)*dx)
            L2_changes.append(abs(L2_noisy - L2_clean)/L2_clean)
            grad_clean = np.gradient(clean, dx)
            grad_noisy = np.gradient(noisy, dx)
            H1_clean = np.sqrt(L2_clean**2 + np.sum(grad_clean**2)*dx)
            H1_noisy = np.sqrt(L2_noisy**2 + np.sum(grad_noisy**2)*dx)
            H1_changes.append(abs(H1_noisy - H1_clean)/H1_clean)
            window_len = 20
            center = 100
            start = max(0, center - window_len//2)
            end = min(len(x), center + window_len//2)
            weights = np.exp(-((x[start:end] - 0.5)**2) / (0.05**2))
            Sw_clean = np.sqrt(np.sum(weights * clean[start:end]**2)*dx)
            Sw_noisy = np.sqrt(np.sum(weights * noisy[start:end]**2)*dx)
            Sw_changes.append(abs(Sw_noisy - Sw_clean)/Sw_clean)
        mean_L2.append(np.mean(L2_changes)*100)
        std_L2.append(np.std(L2_changes)*100)
        mean_H1.append(np.mean(H1_changes)*100)
        std_H1.append(np.std(H1_changes)*100)
        mean_Sw.append(np.mean(Sw_changes)*100)
        std_Sw.append(np.std(Sw_changes)*100)
    return noise_levels, mean_L2, std_L2, mean_H1, std_H1, mean_Sw, std_Sw

noise_lev, mL2, sL2, mH1, sH1, mSw, sSw = compute_relative_changes_with_std()

x = np.arange(len(noise_lev))
width = 0.25
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width, mH1, width, yerr=sH1, capsize=5, label='H^1', color='salmon', error_kw={'ecolor':'darkred'})
ax.bar(x, mL2, width, yerr=sL2, capsize=5, label='L^2', color='skyblue', error_kw={'ecolor':'darkblue'})
ax.bar(x + width, mSw, width, yerr=sSw, capsize=5, label='S_w^2 (ours)', color='lightgreen', error_kw={'ecolor':'darkgreen'})
ax.set_xlabel('Noise level (%)', fontsize=12)
ax.set_ylabel('Relative norm change (%)', fontsize=12)
ax.set_title('Norm sensitivity comparison (1000 realizations, error bars = std)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(noise_lev)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('noise_sensitivity_new.png', dpi=200)
plt.show()
print("New image saved: noise_sensitivity_new.png")
