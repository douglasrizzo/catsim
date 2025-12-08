import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Setup
fig, ax = plt.subplots(figsize=(12, 6))

# Define grade boundaries
boundaries = [-1.5, -0.5, 0.5, 1.5]
grade_names = ['F', 'D', 'C', 'B', 'A']
grade_colors = ['#ff6b6b', '#ffa07a', '#ffd93d', '#95e1d3', '#6bcf7f']

# Define the ability scale range
theta_range = np.linspace(-3, 3, 1000)

# Draw grade regions with shading
regions = [
  (-3, boundaries[0])
] + [
  (boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)
] + [
  (boundaries[-1], 3)
]

for i, (start, end) in enumerate(regions):
    ax.axvspan(start, end, alpha=0.15, color=grade_colors[i], label=f'Grade {grade_names[i]}')
    # Add grade label in the middle of each region
    mid = (start + end) / 2
    ax.text(
        mid, 0.85, grade_names[i], ha='center', va='center',
        fontsize=16, fontweight='bold', color=grade_colors[i],
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor=grade_colors[i],
            linewidth=2
        )
    )

# Draw boundary lines
for boundary in boundaries:
    ax.axvline(boundary, color='black', linestyle='--', linewidth=2, alpha=0.5)

# Example: examinee with theta = 1.0 and SEE = 0.2
# This gives a CI that falls entirely within grade B [0.5, 1.5)
estimated_theta = 1.0
see = 0.2
confidence = 0.90

# Calculate confidence interval
z_score = stats.norm.ppf((1 + confidence) / 2)
ci_lower = estimated_theta - z_score * see
ci_upper = estimated_theta + z_score * see

# Draw normal distribution curve
normal_curve = stats.norm.pdf(theta_range, estimated_theta, see)
# Scale it for visibility
normal_curve_scaled = normal_curve * 0.6 / np.max(normal_curve)

ax.plot(theta_range, normal_curve_scaled, 'b-', linewidth=2.5, label='Ability distribution')

# Shade the confidence interval
mask = (theta_range >= ci_lower) & (theta_range <= ci_upper)
ax.fill_between(
    theta_range[mask], 0, normal_curve_scaled[mask],
    alpha=0.4, color='blue', label=f'{int(confidence*100)}% Confidence Interval'
)

# Mark the estimated theta
ax.plot(estimated_theta, 0, 'ro', markersize=15, label=f'Estimated θ = {estimated_theta}', zorder=5)
ax.axvline(estimated_theta, color='red', linestyle='-', linewidth=2, alpha=0.7, ymax=0.65)

# Mark confidence interval bounds
ax.axvline(ci_lower, color='blue', linestyle=':', linewidth=2, alpha=0.8)
ax.axvline(ci_upper, color='blue', linestyle=':', linewidth=2, alpha=0.8)

# Add arrows and labels for CI bounds
ax.annotate(
    f'CI Lower\n{ci_lower:.2f}', xy=(ci_lower, 0.35), xytext=(ci_lower-0.4, 0.45),
    fontsize=10, ha='center', color='blue',
    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5)
)
ax.annotate(
    f'CI Upper\n{ci_upper:.2f}', xy=(ci_upper, 0.35), xytext=(ci_upper+0.4, 0.45),
    fontsize=10, ha='center', color='blue',
    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5)
)

# Add decision box
ax.text(
    1.0, 0.72, 'CI entirely in Grade B\n→ STOP TEST',
    ha='center', va='center', fontsize=12, fontweight='bold',
    bbox=dict(
        boxstyle='round,pad=0.8',
        facecolor='lightgreen',
        edgecolor='darkgreen',
        linewidth=3
    )
)

# Formatting
ax.set_xlabel('Ability (θ)', fontsize=14, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
ax.set_title('Confidence Interval-Based Stopping Criterion', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 0.95)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.show()