import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


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

def mix_random_with_others(init_choices):
    """
    For a given seed JSON, create hybrid strategies that mix
    Random_seed_X with every other strategy.
    """
    mixed = {}

    # find the random strategy key
    random_key = None
    for k in init_choices:
        if k.startswith("Random_seed_"):
            random_key = k
            break

    if random_key is None:
        print("No Random strategy found — skipping mixing.")
        return init_choices

    random_indices = set(init_choices[random_key])

    # keep originals
    mixed.update(init_choices)

    # create hybrids
    for k, v in init_choices.items():
        if k == random_key:
            continue

        merged_name = f"{random_key}+{k}"
        merged_indices = list(random_indices | set(v))  # union but only with random
        mixed[merged_name] = merged_indices

    return mixed


def normalize_strategy_name(name):
    return re.sub(r"_seed_\d+", "", name)

def compute_average_stopping_per_dataset(results_base_path):

    dataset_results = {}

    for dataset_folder in os.listdir(results_base_path):
        dataset_path = os.path.join(results_base_path, dataset_folder)

        if not os.path.isdir(dataset_path):
            continue

       #print(f"\nProcessing dataset: {dataset_folder}")

        strategy_values = {}

        for seed_folder in os.listdir(dataset_path):
            seed_path = os.path.join(dataset_path, seed_folder)

            csv_path = os.path.join(seed_path, "mae_priors_stopping_indices.csv")

            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                raw_strategy = row["Strategy"]
                stopping_value = pd.to_numeric(row["StoppingIteration"], errors="coerce")

                if pd.isna(stopping_value):
                    continue

                strategy = normalize_strategy_name(raw_strategy)

                strategy_values.setdefault(strategy, []).append(stopping_value)

        avg_results = {
            strategy: sum(values) / len(values)
            for strategy, values in strategy_values.items()
            if len(values) > 0
        }

        dataset_results[dataset_folder] = avg_results

        avg_df = pd.DataFrame.from_dict(
            avg_results,
            orient="index",
            columns=["Average_StoppingIteration"]
        )

        avg_df.index.name = "Strategy"

        output_path = os.path.join(dataset_path, "average_stopping_per_method.csv")
        avg_df.to_csv(output_path)

       #print(f"Saved averages to {output_path}")

    return dataset_results



# Use your existing mappings
# STATIC_NUMBER = 232
# BASE_MEASUREMENTS = 342
# STRATEGY_DISPLAY_NAMES = {...}
def plot_average_stopping_heatmap(
    results_base_path,
    save_path,
    max_iter=100,
    static_number=232,
    base_measurements=342,
    mean_threshold=197
):

   

    # ---------- Normalize strategy name (Random first) ----------
    def normalize_mixed_name(name: str):
        parts = name.split("+")
        parts = [re.sub(r"_seed_\d+", "", p) for p in parts]
        parts = [STRATEGY_DISPLAY_NAMES.get(p, p) for p in parts]

        if "Random" in parts:
            others = sorted([p for p in parts if p != "Random"])
            parts = ["Random"] + others
        else:
            parts = sorted(parts)

        return "+".join(parts)

    rows = []

    # ---------- Collect average CSVs ----------
    for dataset_folder in os.listdir(results_base_path):

        dataset_path = os.path.join(results_base_path, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        avg_csv = os.path.join(dataset_path, "average_stopping_per_method.csv")
        if not os.path.exists(avg_csv):
            continue

        df = pd.read_csv(avg_csv)
        if df.empty:
            continue

        m = re.search(r"(\d+)", dataset_folder)
        folder_id = int(m.group(1)) if m else dataset_folder

        for _, r in df.iterrows():

            strat = str(r["Strategy"])

            # Keep only mixed strategies
            if "+" not in strat:
                continue

            avg_stop = pd.to_numeric(
                r["Average_StoppingIteration"],
                errors="coerce"
            )

            if pd.isna(avg_stop):
                continue

            rows.append({
                "Folder": folder_id,
                "Strategy": normalize_mixed_name(strat),
                "AvgStop": float(avg_stop)
            })

    if not rows:
        print("No valid mixed strategies found.")
        return

    long_df = pd.DataFrame(rows)

    # ---------- Convert stopping → reduction ----------
    long_df["TotalReduction"] = static_number + (
        max_iter - long_df["AvgStop"]
    )

    # ---------- Pivot ----------
    heatmap_data = long_df.pivot_table(
        index="Folder",
        columns="Strategy",
        values="TotalReduction",
        aggfunc="mean"
    ).sort_index()

    if heatmap_data.empty:
        print("No data after pivot.")
        return

    # ---------- Filter by mean threshold ----------
    col_means = heatmap_data.mean(axis=0)
    selected_cols = col_means[
        col_means > mean_threshold
    ].sort_values(ascending=False).index

    heatmap_data = heatmap_data[selected_cols]

    if heatmap_data.empty:
        print("No strategies pass threshold.")
        return

    # Sort columns by performance
    heatmap_data = heatmap_data.loc[
        :, heatmap_data.mean().sort_values(ascending=False).index
    ]

    # ---------- Convert to percentage ----------
    percent_data = heatmap_data / base_measurements * 100

    # Annotation labels (raw reduction numbers)
    annot_labels = heatmap_data.round(0)

    # ---------- Plot ----------
    sns.set_theme(style="white")
    sns.set(font_scale=0.95)

    fig, ax = plt.subplots(
        figsize=(
            max(14, 0.75 * len(heatmap_data.columns)),
            max(6, 0.55 * len(heatmap_data.index))
        )
    )

    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        percent_data,
        cmap=cmap,
        annot=annot_labels,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        square=True,
        vmin=percent_data.min().min(),
        vmax=percent_data.max().max(),
        ax=ax,
        cbar_kws={
            "label": "Total Measurement Reduction (%)",
            "shrink": 0.75
        },
        annot_kws={
            "fontsize": 9,
            "weight": "bold"
        }
    )

    # Labels
    ax.set_ylabel("Materials Library ID", fontsize=13)
    ax.set_xlabel("Initialization Strategy", fontsize=13)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=13
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=13
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # ---------- Save ----------
    output_file = (
        save_path if save_path.endswith(".pdf")
        else save_path + ".pdf"
    )

    plt.savefig(
        output_file,
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_dataset_strategies_bar(
    results_base_path,
    random_csv_path,
    dataset_id,
    best_strategy_csv_path,
    evenly_csv_path,
    save_path,
    max_iter=100,
    static_number=232
):

    # -------- Normalize strategy name ----------
    def normalize_mixed_name(name: str):
        parts = name.split("+")
        parts = [re.sub(r"_seed_\d+", "", p) for p in parts]
        parts = [STRATEGY_DISPLAY_NAMES.get(p, p) for p in parts]

        if "Random" in parts:
            others = sorted([p for p in parts if p != "Random"])
            parts = ["Random"] + others
        else:
            parts = sorted(parts)

        return "+".join(parts)

    # -------- Locate dataset folder ----------
    target_folder = None

    for folder in os.listdir(results_base_path):
        if str(dataset_id) in folder:
            target_folder = os.path.join(results_base_path, folder)
            break

    if target_folder is None:
        print("Dataset folder not found.")
        return

    avg_csv = os.path.join(target_folder, "average_stopping_per_method.csv")
    if not os.path.exists(avg_csv):
        print("average_stopping_per_method.csv not found.")
        return

    df = pd.read_csv(avg_csv)

    # Keep only mixed strategies
    df = df[df["Strategy"].str.contains("+", regex=False)]
    df["AvgStop"] = pd.to_numeric(df["Average_StoppingIteration"], errors="coerce")
    df = df.dropna()

    if df.empty:
        print("No mixed strategies found.")
        return

    # -------- Convert stopping → reduction ----------
    df["Reduction"] = static_number + (max_iter - df["AvgStop"])
    df["Strategy"] = df["Strategy"].apply(normalize_mixed_name)

    df = df.sort_values("Reduction", ascending=False)

    # -------- Load external BEST strategy ----------
    best_df = pd.read_csv(best_strategy_csv_path)
    best_df["Dataset"] = best_df["Dataset"].astype(int)

    best_row = best_df[best_df["Dataset"] == dataset_id]

    external_best_name = None
    external_best_value = None

    if not best_row.empty:
        external_best_name = normalize_mixed_name(
            best_row.iloc[0]["BestStrategy"]
        )
        external_best_value = best_row.iloc[0]["Reduction"]

        # Add it if not already in df
        if external_best_name not in df["Strategy"].values:
            df = pd.concat([
                df,
                pd.DataFrame({
                    "Strategy": [external_best_name],
                    "Reduction": [external_best_value]
                })
            ])
    # -------- Load Random Avg ----------
    random_df = pd.read_csv(random_csv_path)
    random_df.columns = random_df.columns.str.strip()

    # Extract numeric dataset id from folder
    random_df["Dataset"] = (
        random_df["Folder"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    random_row = random_df[random_df["Dataset"] == dataset_id]

    if random_row.empty:
        print(f"Random reduction not found for dataset {dataset_id}")
        print("Available datasets:", random_df["Dataset"].tolist())
        return

    random_reduction = float(random_row["Reduction"].values[0])
    # -------- Add Random 10-seed average ----------
    df = pd.concat([
        df,
        pd.DataFrame({
            "Strategy": ["Random (10 seeds avg)"],
            "Reduction": [random_reduction]
        })
    ])

     # -------- Fixed / Evenly ----------
    evenly_df = pd.read_csv(evenly_csv_path)


    evenly_row = evenly_df[evenly_df["MaterialID"] == dataset_id]

    if not evenly_row.empty:
        stopping = float(evenly_row["EvenlyStoppingIteration"].values[0])
        fixed_reduction = static_number + (max_iter - stopping)

        df = pd.concat([
            df,
            pd.DataFrame({
                "Strategy": ["Fixed (Evenly)"],
                "Reduction": [fixed_reduction]
            })
        ])
    
    df = df.sort_values("Reduction", ascending=False)

    # -------- Plot ----------
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Professional scientific palette
    color_mixed  = "#A6BDD7"   # light desaturated blue
    color_best   = "#C74040"   # deep red
    color_random = "#E8E1E1"
    color_fixed  = "#2E7D32"


    colors = []
    for strat in df["Strategy"]:
        if strat == external_best_name:
            colors.append(color_best)
        elif strat == "Random (10 seeds avg)":
            colors.append(color_random)
        elif strat == "Fixed (Evenly)":
            colors.append(color_fixed)
        else:
            colors.append(color_mixed)

    bars = ax.bar(
        df["Strategy"],
        df["Reduction"],
        color=colors
    )


    # Annotate values
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=16
        )

    # Titles & labels
    #ax.set_title(
     #   f"Dataset {dataset_id} – Strategy Reduction Comparison",
     #   fontsize=14,
     #   pad=15
    #)

    ax.set_ylabel("Total Measurement Reduction", fontsize=16)
    ax.set_xlabel("Initialization Strategy", fontsize=16)

    plt.xticks(rotation=45, ha="right", fontsize=16)

    # Light grid (professional look)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Remove top/right spines only
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend (clean and minimal)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_mixed, edgecolor="none", label="Mixed Strategies"),
        Patch(facecolor=color_best, edgecolor="none", label="Best Strategy"),
        Patch(facecolor=color_random, edgecolor="none", label="Random (10 seeds avg)"),
        Patch(facecolor=color_fixed, edgecolor="none", label="Fixed Grid")
    ]

    ax.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def compute_random_reduction(
    base_path,
    max_iteration=100,
    static_number=232,
    output_filename="strategy_reduction_results.csv"
):
    results = []

    for dataset_folder in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_folder)

        if os.path.isdir(dataset_path):
            file_path = os.path.join(dataset_path, "mae_priors_stopping_indices.csv")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()

                if "StoppingIteration" not in df.columns:
                    print(f"⚠ Skipping {dataset_folder} (StoppingIteration column missing)")
                    continue

                avg_stopping = df["StoppingIteration"].mean()
                reduction = (max_iteration - avg_stopping) + static_number

                results.append({
                    "Folder": dataset_folder,
                    "AvgStopping": avg_stopping,
                    "Reduction": reduction
                })

                #print(f"{dataset_folder} | AvgStopping: {avg_stopping:.2f} | Reduction: {reduction:.2f}")

    if not results:
        print("No valid datasets found.")
        return None

    results_df = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(base_path, output_filename)
    results_df.to_csv(csv_path, index=False)


    return results_df

