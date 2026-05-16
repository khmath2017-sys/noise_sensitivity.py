import numpy as np
import matplotlib.pyplot as plt

noise_levels = [0, 5, 10]
rel_change_H1 = [0, 450, 780]
rel_change_L2 = [0, 2, 4]
rel_change_Sw = [0, 0.3, 0.6]

x = np.arange(len(noise_levels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width, rel_change_H1, width, label='H^1', color='salmon')
ax.bar(x, rel_change_L2, width, label='L^2', color='skyblue')
ax.bar(x + width, rel_change_Sw, width, label='S_w^2 (ours)', color='lightgreen')

ax.set_xlabel('Noise level (%)')
ax.set_ylabel('Relative norm change (%)')
ax.set_title('Norm sensitivity comparison under noise')
ax.set_xticks(x)
ax.set_xticklabels(noise_levels)
ax.legend()
ax.grid(True, axis='y')
plt.savefig('noise_sensitivity_comparison.png', dpi=150)
plt.show()
print("Image saved as noise_sensitivity_comparison.png")
