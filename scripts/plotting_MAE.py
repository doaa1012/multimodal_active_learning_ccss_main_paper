import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import itertools
import matplotlib.colors as mcolors
import re
import numpy as np



STRATEGY_DISPLAY_NAMES = {
    "Centroids_saturation_high": "Cent_sat_high",
    "Centroids_saturation_medium": "Cent_sat_med",
    "Centroids_saturation_low": "Cent_sat_low",
    "Top5Similarity": "T5S",
    "Max Comp": "Max Comp",
    "Min Comp": "Min Comp",
    "Random": "Random",
    "LHS": "LHS",
    "K-Means": "K-Means",
    "Farthest": "FPS",
    "K-Center": "K-Center",
    "ODAL": "ODAL"
}
base_strategies = [
    "Top5Similarity", "Max Comp", "Min Comp", 
    "Centroids_saturation_high", "Centroids_saturation_medium", "Centroids_saturation_low",
    "Random", "LHS", "K-Means", "Farthest", "K-Center", "ODAL"
]
def plot_strategy_across_datasets(
    strategy,
    dataset_paths,
    dataset_labels,
    save_path=None,
    measurement_uncertainty=0.005,
    title=None
):
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    iterations = list(range(100))

    for i, path in enumerate(dataset_paths):
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            continue

        df = pd.read_csv(path)
        if strategy not in df.columns:
            print(f"[Warning] Strategy '{strategy}' not found in: {path}")
            continue

        raw_values = df[strategy]
        interpolated = raw_values.interpolate(limit_direction='both')
        values = interpolated.mask(raw_values.isna()).values[:100]

        color = distinct_colors[i % len(distinct_colors)]
        label = dataset_labels[i] if i < len(dataset_labels) else f"Dataset {i+1}"

        ax.plot(
            iterations[:len(values)],
            values,
            label=label,
            color=color,
            linestyle='-',
            linewidth=1.5
        )

    ax.axhline(
        y=measurement_uncertainty,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Measurement Uncertainty'
    )

    # Axis labels (larger font)
    ax.set_xlabel("Iteration", fontsize=15, labelpad=8)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=15, labelpad=8)

    # Tick labels (larger font)
    ax.tick_params(axis='both', labelsize=13)

    # Explanatory note below x-label
    ax.text(
        0.5, -0.18,
        "Total number of measurements = Iteration + 5 (Initial measurements)",
        transform=ax.transAxes,
        fontsize=11,
        ha='center',
        va='top'
    )

    plt.subplots_adjust(bottom=0.18)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 100])

    if title:
        ax.set_title(title, fontsize=16, pad=14)

    # Legend with bigger font
    ax.legend(
        title="Dataset",
        fontsize=12,          # legend entries
        title_fontsize=13,    # legend title
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=True,
        handletextpad=0.4,
        columnspacing=1.2
    )

    plt.tight_layout()

    if save_path:
        if not save_path.lower().endswith(".pdf"):
            save_path = save_path.rsplit('.', 1)[0] + ".pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


def get_large_color_palette(n):
    """Return n distinct colors by combining colormaps."""
    cmap_list = ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set3', 'Paired', 'Pastel1']
    color_list = []

    for cmap_name in cmap_list:
        cmap = cm.get_cmap(cmap_name)
        for i in range(cmap.N):
            rgba = cmap(i)
            color_list.append(mcolors.to_hex(rgba))
            if len(color_list) >= n:
                return color_list
    return color_list[:n]





def plot_all_base_and_mixed_strategies(df, main_strategy, base_strategies, save_path=None, measurement_uncertainty=0.005):
    strategies_to_plot = []
    labels = []
    styles = []

    full_strategy_list = []

    for base in base_strategies:
        full_strategy_list.append(base)
        full_strategy_list.append(f"{main_strategy}+{base}")

    color_palette = get_large_color_palette(len(full_strategy_list))
    color_map = dict(zip(full_strategy_list, color_palette))

    for base in base_strategies:
        base_label = STRATEGY_DISPLAY_NAMES.get(base, base)
        mixed_label = f"{STRATEGY_DISPLAY_NAMES.get(main_strategy, main_strategy)}+{base_label}"

        if base in df.columns:
            strategies_to_plot.append(base)
            labels.append(base_label)

            # Special style for even_space
            if base == "even_space":
                styles.append(("solid", "#d62728"))  # bold red
            else:
                styles.append(("solid", color_map[base]))

        mixed_name = f"{main_strategy}+{base}"
        if mixed_name in df.columns:
            strategies_to_plot.append(mixed_name)
            labels.append(mixed_label)
            styles.append(("dashed", color_map[mixed_name]))

    if not strategies_to_plot:
        print("No matching strategies found in the data.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = list(range(100))

    for strategy, label, (linestyle, color) in zip(strategies_to_plot, labels, styles):
        raw_values = df[strategy] if strategy in df.columns else pd.Series([None]*100)
        interpolated = raw_values.interpolate(limit_direction='both')
        values = interpolated.mask(raw_values.isna()).values[:100]

        is_even_space = (strategy == "even_space")

        ax.plot(
            iterations[:len(values)],
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=3.5 if is_even_space else 2,
            marker='o' if is_even_space else None,
            markersize=4 if is_even_space else 0,
            zorder=10 if is_even_space else 5
        )

    # Measurement uncertainty line
    ax.axhline(
        y=measurement_uncertainty,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label='Measurement Uncertainty'
    )

    # Axis, labels, legend
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 100])

    ax.legend(
        title="Strategy",
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=True,
        handletextpad=0.4,
        columnspacing=1.2
    )

    plt.tight_layout()

    if save_path:
        if not save_path.endswith(".pdf"):
            save_path = save_path.rsplit('.', 1)[0] + ".pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
        #print(f"Saved plot to: {save_path}")
    else:
        plt.show()
import pandas as pd
import matplotlib.pyplot as plt

def plot_initialization_strategies(
    csv_path,
    all_init_strategies,
    strategy_order=None,
    strategy_rename=None,   # <-- added here
    resistance_col="Resistance",
    x_col="x",
    y_col="y",
    max_points=342,
    output_path=None,
):
    # Load data
    data = pd.read_csv(csv_path).iloc[:max_points]
    x = data[x_col].values
    y = data[y_col].values
    resistance = data[resistance_col].values

    cmap = plt.colormaps["plasma"]

    # Use given order or default to dict order
    if strategy_order is None:
        strategy_order = list(all_init_strategies.keys())

    # If rename list provided, create mapping
    rename_map = None
    if strategy_rename and len(strategy_rename) == len(strategy_order):
        rename_map = dict(zip(strategy_order, strategy_rename))

    num_strategies = len(strategy_order)
    cols = 6
    rows = 2

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3, rows * 4))
    axes = axes.flatten()

    vmin, vmax = resistance.min(), resistance.max()
    scatter = None  # For the colorbar

    for idx, strategy in enumerate(strategy_order):
        if idx >= len(axes):
            break
        indices = all_init_strategies[strategy]
        ax = axes[idx]
        ax.set_aspect("equal")

        # Base scatter plot
        scatter = ax.scatter(
            x, y, c=resistance, cmap=cmap, marker="s", s=50, vmin=vmin, vmax=vmax
        )

        # Highlight selected indices
        ax.scatter(
            x[indices], y[indices],
            c="white", marker="X", s=200, edgecolor="black", linewidth=2
        )
        ax.scatter(
            x[indices], y[indices],
            c="red", marker="o", s=100, edgecolor="black", linewidth=1, alpha=0.8
        )

        # Label selected points
        for i in indices:
            ax.text(
                x[i], y[i], 'X', fontsize=12, color='black',
                ha='center', va='center', fontweight='bold'
            )

        ax.set_xticks([])
        ax.set_yticks([])

        # ✅ Use short name if provided
        title = rename_map.get(strategy, strategy) if rename_map else strategy
        ax.set_title(title, fontsize=14)

    # Hide unused subplots
    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    # Add colorbar inside the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Resistance (Ohm)", fontsize=12)

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches='tight', dpi=300)
    else:
        plt.show()
