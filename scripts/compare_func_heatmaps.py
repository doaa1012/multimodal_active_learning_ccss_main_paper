import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import warnings
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)

# The experiment begins with 5 to 10 initial measurements and runs for 100 iterations.
# This leaves 232 unmeasured points on the wafer grid.
STATIC_NUMBER = 232 

BASE_STRATEGIES = [
    "Top5Similarity", "Max Comp", "Min Comp", 
    "Centroids_saturation_high", "Random", "LHS", 
    "K-Means", "Farthest", "K-Center", "ODAL", 
    "Centroids_saturation_medium", "Centroids_saturation_low"
]

# Strategy display name mapping
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
def detect_base_strategy(strategy, base_list):
    parts = strategy.split('+')
    if len(parts) < 2:
        return None
    if parts[0] in base_list:
        return parts[0]
    return None

def analyze_stopping_iteration_differences(base_dir, save_csv_path, save_plot_path):
    comparison_results = []

    for folder in os.listdir(base_dir):
        if folder.endswith("_results"):
            file_path = os.path.join(base_dir, folder, "mae_priors_stopping_indices.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                if 'Strategy' not in df.columns or 'StoppingIteration' not in df.columns:
                    continue

                base_values = {
                    row['Strategy']: row['StoppingIteration']
                    for _, row in df.iterrows()
                    if row['Strategy'] in BASE_STRATEGIES
                }

                for _, row in df.iterrows():
                    strategy = row['Strategy']
                    #if "Centroids_saturation_medium" in strategy or "Centroids_saturation_low" in strategy:
                    #    continue
                    stopping_value = row['StoppingIteration']
                    base = detect_base_strategy(strategy, BASE_STRATEGIES)

                    if base and strategy != base and base in base_values:
                        base_value = base_values[base]
                        comparison_results.append({
                            "Folder": folder,
                            "Base": base,
                            "Mixed": strategy,
                            "BaseValue": base_value,
                            "MixedValue": stopping_value,
                            "Difference": stopping_value - base_value
                        })

    # Create DataFrame
    result_df = pd.DataFrame(comparison_results)

    # Save CSV
    result_df.to_csv(save_csv_path, index=False)
    #print(f"Comparison saved to '{save_csv_path}'")

    # Classify changes
    result_df["Change"] = result_df["Difference"].apply(
        lambda x: "Increased" if x > 0 else "Decreased" if x < 0 else "No Change"
    )

    # Group counts
    change_counts = result_df.groupby(["Base", "Change"]).size().unstack(fill_value=0)
    change_counts = change_counts.reindex(BASE_STRATEGIES, fill_value=0)
    change_counts.index = [STRATEGY_DISPLAY_NAMES.get(name, name) for name in change_counts.index]

    change_counts = change_counts[["Decreased", "Increased", "No Change"]]

    # Plot with annotations
    colors = {
        "Decreased": "#EF4444",  
        "Increased": "#34D399",  
        "No Change": "#3B82F6"   
    }

    ax = change_counts.plot(
        kind="bar",
        stacked=True,
        color=[colors[col] for col in change_counts.columns],
        figsize=(18, 6)
    )

    # Annotate bars
    for i, base in enumerate(change_counts.index):
        y_offset = 0
        for col in ["Decreased", "Increased", "No Change"]:
            value = change_counts.loc[base, col]
            if value > 0:
                ax.text(i, y_offset + value / 2, str(value), ha='center', va='center', fontsize=10, color='white')
                y_offset += value

    #plt.title("Change in StoppingIteration Value by Base Strategy (Mixed Combinations)", fontsize=16)
    plt.ylabel("Number of Mixed Strategy Comparisons", fontsize=14)
    plt.xlabel("Base Strategy", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Change", fontsize=11, title_fontsize=12)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_plot_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

    #print(f"Plot saved to '{save_plot_path}'")
    plt.close()

def plot_decreased_only_heatmap_sorted(comparison_df, save_path):
    if comparison_df.empty:
        print("No data to plot.")
        return

    # Filter for decreased only
    decreased_df = comparison_df[comparison_df["Difference"] < 0].copy()
    if decreased_df.empty:
        print("No decreased values to plot.")
        return

    # Extract numeric object ID and calculate absolute difference
    decreased_df["Folder"] = decreased_df["Folder"].str.extract(r"(\d+)").astype(int)
    STATIC_NUMBER = 232
    BASE_MEASUREMENTS = 342  # reference for percentage calculation
    decreased_df["AbsDifference"] = decreased_df["Difference"].abs() + STATIC_NUMBER

    # Normalize strategy names
    def normalize_strategy_name(name):
        parts = name.split('+')
        normalized = '+'.join(sorted(parts))
        display_parts = [STRATEGY_DISPLAY_NAMES.get(p, p) for p in normalized.split('+')]
        return '+'.join(display_parts)

    decreased_df["Mixed"] = decreased_df["Mixed"].apply(normalize_strategy_name)

    # Create pivot table
    heatmap_data = decreased_df.pivot_table(
        index="Folder",
        columns="Mixed",
        values="AbsDifference",
        aggfunc="mean"
    )

    # Compute column averages and filter
    column_means = heatmap_data.fillna(0).mean(axis=0)
    selected_columns = column_means[column_means > 162].sort_values(ascending=False).index
    heatmap_data = heatmap_data[selected_columns]

    annot_labels = heatmap_data.map(lambda x: f"{int(x)}" if pd.notnull(x) else "")

    # Save rankings
    strategy_ranking = column_means.sort_values(ascending=False)
    csv_path = os.path.join(os.path.dirname(save_path), "strategy_reduction_ranking.csv")
    strategy_ranking.to_csv(csv_path, header=["AvgReduction"])

    if heatmap_data.empty:
        print("No strategies with average reduction over threshold.")
        return

    sns.set_theme(style="whitegrid")
    sns.set(font_scale=0.8)

    fig, ax = plt.subplots(
        figsize=(max(20, 0.6 * len(heatmap_data.columns)),
                 max(6, 0.4 * len(heatmap_data.index)))
    )

    sns.heatmap(
        heatmap_data,
        cmap="Reds",
        annot=annot_labels,
        fmt="",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        ax=ax,
        cbar_kws={
            "label": "Total Measurements Reduction",
            "shrink": 0.5,
            "aspect": 10,
            "pad": 0.01
        }
    )

    # Axis labels
    ax.set_ylabel("Materials Library ID", fontsize=12.5)
    ax.set_xlabel("Initialization Strategy", fontsize=15)

    # Tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    ticks = cbar.get_ticks()
    ticklabels = []
    for t in ticks:
        perc = 100 * t / BASE_MEASUREMENTS
        ticklabels.append(f"{int(t)} ({perc:.0f}%)")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.set_ylabel("Total Measurement Reduction", fontsize=13)

    # Remove axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(right=0.88, bottom=0.3)

    # Save
    plt.savefig(save_path.replace('.png', '.pdf'),
                format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

 
def plot_heatmap_base_less_than_100(comparison_df, save_path):
    if comparison_df.empty:
        print("No data to plot.")
        return

    # Filter for BaseValue < 100
    filtered_df = comparison_df[comparison_df["BaseValue"] < 100].copy()
    if filtered_df.empty:
        print("No entries with BaseValue < 100.")
        return

    # Compute Difference and absolute value
    STATIC_NUMBER = 232
    BASE_MEASUREMENTS = 342  # reference for percentages
    filtered_df["Difference"] = 100 - filtered_df["BaseValue"]
    filtered_df["AbsDifference"] = filtered_df["Difference"] + STATIC_NUMBER

    # Extract numeric folder ID
    filtered_df["Folder"] = filtered_df["Folder"].str.extract(r"(\d+)").astype(int)

    # Normalize strategy names from Base column
    def normalize_strategy_name(name):
        parts = name.split('+')
        normalized = '+'.join(sorted(parts))
        display_parts = [STRATEGY_DISPLAY_NAMES.get(p, p) for p in normalized.split('+')]
        return '+'.join(display_parts)

    filtered_df["Base"] = filtered_df["Base"].apply(normalize_strategy_name)

    # Pivot to form heatmap
    heatmap_data = filtered_df.pivot_table(
        index="Folder",
        columns="Base",
        values="AbsDifference",
        aggfunc="mean"
    )

    # Filter strategies with meaningful reduction
    column_means = heatmap_data.fillna(0).mean(axis=0)
    csv_path = os.path.join(os.path.dirname(save_path), "base_reduction_mean_ranking.csv")
    column_means.sort_values(ascending=False).to_csv(csv_path, header=["MeanReduction"])
    selected_columns = column_means[column_means > 75].sort_values(ascending=False).index
    heatmap_data = heatmap_data[selected_columns]

    # Prepare annotation labels (numbers only)
    annot_labels = heatmap_data.map(lambda x: f"{int(x)}" if pd.notnull(x) else "")

    if heatmap_data.empty:
        print("No strategies with sufficient reduction.")
        return

    # Plot
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=0.8)

    fig, ax = plt.subplots(
        figsize=(max(20, 0.6 * len(heatmap_data.columns)), 
                 max(6, 0.4 * len(heatmap_data.index)))
    )

    sns.heatmap(
        heatmap_data,
        cmap="Blues",
        annot=annot_labels,
        fmt="",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        ax=ax,
        cbar_kws={
            "label": "Total Measurements Reduction",
            "shrink": 0.5,
            "aspect": 10,
            "pad": 0.01
        }
    )

    # Axis labels
    ax.set_ylabel("Materials Library ID", fontsize=12)
    ax.set_xlabel("Initialization Strategy", fontsize=14)

    # Tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    # Customize colorbar to add percentage beside numbers
    cbar = ax.collections[0].colorbar
    ticks = cbar.get_ticks()
    ticklabels = []
    for t in ticks:
        perc = 100 * t / BASE_MEASUREMENTS
        ticklabels.append(f"{int(t)} ({perc:.0f}%)")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.set_ylabel("Total Measurements Reduction", fontsize=13)

    # Clean up spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(right=0.88, bottom=0.3)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()




def load_and_process_summary(path, label):
    df = pd.read_csv(path)
    df["Acquisition function"] = label
    df["Change"] = df["Difference"].apply(
        lambda x: "Increased" if x > 0 else "Decreased" if x < 0 else "No Change"
    )
    return df

def plot_strategy_comparison_summary(
    sawei_csv_path,
    uncertainty_csv_path,
    output_dir,
    filename_base="strategy_comparison_plot"
):
    # Font and style
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.linewidth'] = 1.2

    # Load datasets
    df_sawei = load_and_process_summary(sawei_csv_path, "SAWEI")
    df_uncertainty = load_and_process_summary(uncertainty_csv_path, "Uncertainty")

    df_all = pd.concat([df_sawei, df_uncertainty], ignore_index=True)

    # Update here: group by 'Acquisition function' instead of 'Source'
    grouped = df_all.groupby(["Base", "Acquisition function", "Change"]).size().unstack(fill_value=0).reset_index()
    acquisition_funcs = ["SAWEI", "Uncertainty"]
    changes = ["Decreased", "Increased", "No Change"]

    # Prepare plot data
    plot_data = []
    for af in acquisition_funcs:
        sub_data = grouped[grouped["Acquisition function"] == af].set_index("Base").reindex(BASE_STRATEGIES, fill_value=0)
        sub_data = sub_data[changes] if all(col in sub_data.columns for col in changes) else pd.DataFrame(columns=changes, index=BASE_STRATEGIES).fillna(0)
        plot_data.append(sub_data)

    # Sort bars
    total_decrease = (
        plot_data[0]["Decreased"].add(plot_data[1]["Decreased"], fill_value=0)
        .sort_values(ascending=False)
    )
    sorted_bases = total_decrease.index.tolist()
    xtick_labels = [STRATEGY_DISPLAY_NAMES.get(base, base) for base in sorted_bases]
    index = range(len(sorted_bases))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bar_width = 0.35
    spacing = 0.15

    colors = {
        "Decreased": "#66c2a5",
        "Increased": "#fc8d62",
        "No Change": "#8da0cb"
    }

    for i, (data, label) in enumerate(zip(plot_data, acquisition_funcs)):
        bottom = [0] * len(sorted_bases)
        for change in changes:
            values = data.loc[sorted_bases][change].values
            positions = [x + i * (bar_width + spacing) for x in index]
            ax.bar(
                positions, values, bar_width,
                bottom=bottom, color=colors[change],
                edgecolor='white', linewidth=0.5,
                hatch='///' if label == "Uncertainty" else '',
                label=f"{change}" if i == 0 else None
            )
            bottom = [sum(x) for x in zip(bottom, values)]

    # Axes and labels
    ax.set_xlabel("Base Strategy", fontsize=14)
    ax.set_ylabel("Accumulative Strategy Pairing", fontsize=14)
    ax.set_xticks([x + (bar_width + spacing) / 2 for x in index])
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legends
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, title="Change Type", fontsize=12, title_fontsize=16,
                        bbox_to_anchor=(1.005, 1), loc='upper left', frameon=False)

    af_legend_handles = [
        Patch(facecolor='gray', edgecolor='white', label='SAWEI', hatch=''),
        Patch(facecolor='gray', edgecolor='white', label='Uncertainty', hatch='///')
    ]
    legend2 = ax.legend(handles=af_legend_handles, title="Acquisition Function", fontsize=12, title_fontsize=16,
                        bbox_to_anchor=(1.005, 0.7), loc='upper left', frameon=False)
    ax.add_artist(legend1)

    plt.tight_layout()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    full_base_path = os.path.join(output_dir, filename_base)
    for ext in ['.png', '.pdf']:
        save_path = f"{full_base_path}{ext}"
        plt.savefig(save_path, format=ext[1:], dpi=300, bbox_inches='tight')

    plt.show()
