import os
import pickle
import numpy as np
import pandas as pd
import re
import pandas as pd
import matplotlib.pyplot as plt

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

def compute_mean_curve_per_dataset(base_path):

    for folder in os.listdir(base_path):
        dataset_path = os.path.join(base_path, folder)

        if os.path.isdir(dataset_path):
            pkl_path = os.path.join(dataset_path, "mae_priors_all_results.pkl")

            if os.path.exists(pkl_path):

                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                mae_dict = data["mae"]

                # Convert dict → matrix (seeds × iterations)
                curves = list(mae_dict.values())
                max_len = max(len(c) for c in curves)

                mat = np.full((len(curves), max_len), np.nan)

                for i, c in enumerate(curves):
                    mat[i, :len(c)] = c

                #  Mean curve across seeds
                mean_curve = np.nanmean(mat, axis=0)

                # Save mean curve
                mean_df = pd.DataFrame({
                    "Iteration": np.arange(len(mean_curve)),
                    "MeanMAE": mean_curve
                })

                output_path = os.path.join(dataset_path, "mean_mae_curve.csv")
                mean_df.to_csv(output_path, index=False)



def normalize_column_name(col):
    # Remove "_seed_XX" everywhere in string
    col = re.sub(r"_seed_\d+", "", col)
    return col

    
def average_dataset_across_seeds(
    dataset_path,
    filename="mae_priors_results.csv",
    output_filename="average_mae_per_method.csv"
):
    """
    Average MAE curves across seeds using the SAME logic as:
        mae_dict_to_matrix()

    Guarantees consistency with seed visualization pipeline.
    """

    collected = {}

    # -----------------------------
    # Collect curves per strategy
    # -----------------------------
    for seed_folder in sorted(os.listdir(dataset_path)):
        seed_path = os.path.join(dataset_path, seed_folder)

        if not os.path.isdir(seed_path):
            continue

        file_path = os.path.join(seed_path, filename)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        for col in df.columns:
            if col.lower() == "iteration":
                continue

            base_name = normalize_column_name(col)
            values = df[col].astype(float).values

            if values.size == 0:
                continue

            collected.setdefault(base_name, []).append(values)

    if not collected:
        return

    mean_results = {}

    # -----------------------------
    # EXACT same logic as mae_dict_to_matrix
    # -----------------------------
    for strategy, curves in collected.items():

        curves = list(curves)
        max_len = max(len(c) for c in curves)

        # seed × iteration matrix
        mat = np.full((len(curves), max_len), np.nan)

        for i, c in enumerate(curves):
            mat[i, :len(c)] = c

        # identical averaging step
        mean_curve = np.nanmean(mat, axis=0)

        mean_results[strategy] = mean_curve

    # -----------------------------
    # Save result
    # -----------------------------
    result_df = pd.DataFrame(mean_results)
    result_df.insert(0, "Iteration", np.arange(len(result_df)))

    save_path = os.path.join(dataset_path, output_filename)
    result_df.to_csv(save_path, index=False)
def average_all_datasets(base_path,
                         filename="mae_priors_results.csv",
                         output_filename="average_mae_per_method.csv"):
    
    #print(f"\nScanning base folder: {base_path}")

    for dataset_folder in sorted(os.listdir(base_path)):
        dataset_path = os.path.join(base_path, dataset_folder)

        if not os.path.isdir(dataset_path):
            continue

        #print(f"\n====================================")
        #print(f"Processing dataset: {dataset_folder}")

        average_dataset_across_seeds(
            dataset_path,
            filename=filename,
            output_filename=output_filename
        )

    #print("\n All datasets processed.")



def _normalize_name(s: str) -> str:
    """Normalize for matching columns across small formatting differences."""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    s = s.lower()
    return s


def to_display(name: str) -> str:
    """Internal -> display (fallback to itself)."""
    name = str(name).strip()
    return STRATEGY_DISPLAY_NAMES.get(name, name)


def to_internal(display_or_internal: str) -> str:
    """Display -> internal (fallback to itself)."""
    # reverse mapping display -> internal
    display_to_internal = {v: k for k, v in STRATEGY_DISPLAY_NAMES.items()}

    s = str(display_or_internal).strip()
    return display_to_internal.get(s, s)


def convert_pair_display_to_internal(best_display: str) -> str:
    """
    'Min Comp+T5S' -> 'Min Comp+Top5Similarity'
    Works even if one part is already internal.
    """
    parts = [p.strip() for p in str(best_display).split("+")]
    if len(parts) != 2:
        raise ValueError(f"BestStrategy must be 'A+B'. Got: {best_display}")

    left = to_internal(parts[0])
    right = to_internal(parts[1])
    return f"{left}+{right}"


def pair_label_from_internal(pair_internal: str) -> str:
    """'Min Comp+Top5Similarity' -> 'Min Comp + T5S' (display label)."""
    parts = [p.strip() for p in str(pair_internal).split("+")]
    if len(parts) != 2:
        return str(pair_internal)
    return f"{to_display(parts[0])} + {to_display(parts[1])}"


def find_best_column(best_mae_df: pd.DataFrame, best_internal: str) -> str:
    """
    Find matching column in best_mae_df:
    - exact match first
    - then normalized match
    """
    cols = list(best_mae_df.columns)

    if best_internal in cols:
        return best_internal

    # Try normalized match
    target_norm = _normalize_name(best_internal)
    norm_map = {_normalize_name(c): c for c in cols}

    if target_norm in norm_map:
        return norm_map[target_norm]

    # If still not found, try "contains" normalized
    for c in cols:
        if target_norm in _normalize_name(c):
            return c

    raise KeyError(f"Best column not found in best_mae_file for: {best_internal}")


def plot_mixed_random_fixed(
        dataset_id,
        seeds_base_path,
        random10_base_path,
        fixed_file,
        best_strategy_file,
        best_mae_file,
        save_path,
        measurement_uncertainty=0.005,
       
        figsize=(9, 6),
        dpi=300):

    dataset_id = int(dataset_id)
    dataset_folder = f"{dataset_id}_results"

    # -----------------------------
    # 1) Mixed strategies averaged (background)
    # -----------------------------
    avg_file = os.path.join(seeds_base_path, dataset_folder, "average_mae_per_method.csv")
    if not os.path.exists(avg_file):
        print(f"Mixed strategies file not found:\n{avg_file}")
        return

    df = pd.read_csv(avg_file)
    df.columns = df.columns.str.strip()

    if "Iteration" not in df.columns:
        print("Expected 'Iteration' column in average_mae_per_method.csv")
        return

    mixed_cols = [c for c in df.columns if "+" in c]
    if not mixed_cols:
        print("No mixed strategies found in average file.")
        return

    # -----------------------------
    # 2) Random-10
    # -----------------------------
    random10_file = os.path.join(random10_base_path, dataset_folder, "mean_mae_curve.csv")
    if not os.path.exists(random10_file):
        print(f"Random-10 file not found:\n{random10_file}")
        return
    random10_df = pd.read_csv(random10_file)
    random10_df.columns = random10_df.columns.str.strip()

    # -----------------------------
    # 3) Evenly spaced
    # -----------------------------
    if not os.path.exists(fixed_file):
        print(f"Fixed baseline file not found:\n{fixed_file}")
        return
    fixed_df = pd.read_csv(fixed_file)
    fixed_df.columns = fixed_df.columns.str.strip()

    # -----------------------------
    # 4) Get best strategy (display) -> internal
    # -----------------------------
    best_df = pd.read_csv(best_strategy_file)
    best_df.columns = best_df.columns.str.strip()

    best_row = best_df[best_df["Dataset"] == dataset_id]
    if best_row.empty:
        print(f"No best strategy found for dataset {dataset_id}")
        return

    best_display = str(best_row["BestStrategy"].values[0]).strip()
    best_internal = convert_pair_display_to_internal(best_display)  # e.g., Min Comp+Top5Similarity
    best_label = pair_label_from_internal(best_internal)           # e.g., Min Comp + T5S

    print("best strategy:", best_display, "->", best_internal)

    # -----------------------------
    # 5) Load best MAE curve file (NO Iteration column; row index is iteration)
    # -----------------------------
    if not os.path.exists(best_mae_file):
        print(f"Best MAE file not found:\n{best_mae_file}")
        return

    best_mae_df = pd.read_csv(best_mae_file)
    best_mae_df.columns = best_mae_df.columns.str.strip()

    try:
        best_col = find_best_column(best_mae_df, best_internal)
    except KeyError as e:
        print(str(e))
        return

    best_curve = best_mae_df[best_col]
    best_iter = np.arange(len(best_curve))  # row index = iteration

    # -----------------------------
    # Plot styling (publication-ish)
    # -----------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Use tab20 but cycle if too many
    cmap = plt.cm.get_cmap("tab20", max(20, len(mixed_cols)))
    colors = [cmap(i % cmap.N) for i in range(len(mixed_cols))]

    # -----------------------------
    # Mixed strategies (ALL, with legend labels)
    # -----------------------------
    for color, col in zip(colors, mixed_cols):
        # legend label = display(displayA) + display(displayB)
        col_label = pair_label_from_internal(col)

        # Make non-best curves thinner + slightly transparent
        lw = 1.6
        alpha = 0.85

        # If this curve corresponds to best_internal (sometimes avg file also has it), keep normal
        if _normalize_name(col) == _normalize_name(best_internal):
            lw = 2.2
            alpha = 0.95

        ax.plot(df["Iteration"], df[col], color=color, linewidth=lw, alpha=alpha, label=col_label, zorder=2)

    # -----------------------------
    # Best curve (from best_mae_file) highlighted
    # -----------------------------
    ax.plot(best_iter, best_curve, color="crimson", linewidth=3.2,
            label=f"Best: {best_label}", zorder=6)

    # -----------------------------
    # Baselines + uncertainty
    # -----------------------------
    if "Iteration" in random10_df.columns and "MeanMAE" in random10_df.columns:
        ax.plot(random10_df["Iteration"], random10_df["MeanMAE"],
                color="black", linestyle="--", linewidth=2.4, label="Random-10 (Mean)", zorder=4)

    if "Iteration" in fixed_df.columns and "MAE" in fixed_df.columns:
        ax.plot(fixed_df["Iteration"], fixed_df["MAE"],
                color="dimgray", linestyle=":", linewidth=2.4, label="Fixed Grid", zorder=4)

    ax.axhline(measurement_uncertainty, color="black", linestyle="-.", linewidth=1.4,
               label="Measurement Uncertainty", zorder=3)

    # -----------------------------
    # Axes formatting
    # -----------------------------
    ax.set_xlabel("Iteration", fontsize=14)

    ax.text(
        0.5, -0.18,
        "Total number of measurements = Iteration + 10 (Initial measurements)",
        transform=ax.transAxes,
        fontsize=14,
        ha='center',
        va='top'
    )

    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend INSIDE, includes EVERYTHING
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.9,
        handlelength=2.5,
        columnspacing=0.9,
        borderpad=0.6,
        fontsize=14   
    )


    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

 

