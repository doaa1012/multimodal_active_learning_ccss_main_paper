import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.measurement_devices import *
from scripts.gaussian_process_sawei import GPSawei
from scripts.run_active_learning import *

def select_paper_style_points(df, x_col='x', y_col='y', inner_step=6):
    """
    Select 9 points like the paper:
      - 1 center
      - 4 inner points around center (offset by `inner_step` grid positions)
      - 4 mid-side outer points
    """
    # Make sorted unique coordinate grids
    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())
    
    # center grid index
    cx = len(x_vals)//2
    cy = len(y_vals)//2

    # index positions for the 4 inner points
    inner_positions = [
        (cx - inner_step, cy - inner_step),
        (cx - inner_step, cy + inner_step),
        (cx + inner_step, cy - inner_step),
        (cx + inner_step, cy + inner_step),
    ]

    # center and mid-side positions
    center_pos = (cx, cy)
    side_positions = [
        (0, cy),
        (len(x_vals)-1, cy),
        (cx, 0),
        (cx, len(y_vals)-1),
    ]

    # Combine all 9 target coordinates in (x,y)
    def find_nearest_idx(x_target, y_target):
        d = (df[x_col]-x_target)**2 + (df[y_col]-y_target)**2
        return int(d.idxmin())

    all_targets = [center_pos] + inner_positions + side_positions
    selected_indices = []
    for xi, yi in all_targets:
        xi = np.clip(xi, 0, len(x_vals)-1)
        yi = np.clip(yi, 0, len(y_vals)-1)
        x_target = x_vals[xi]
        y_target = y_vals[yi]
        selected_indices.append(find_nearest_idx(x_target, y_target))

    return list(set(selected_indices))



def run_active_learning_multiple_gp(
    dataset_path,
    init_indices,
    target_col,
    excluded_cols,
    max_iter,
    output_base_path_dict,
    gp_class_dict,
    summary_results_dict=None
):
    """
    Run active learning on a single dataset using multiple GP models.

    Parameters:
    - dataset_path (str): Path to the dataset CSV file
    - init_indices (list): Initial selected indices
    - target_col (str): Column to predict (e.g., "Resistance")
    - excluded_cols (list): Columns to exclude from features
    - max_iter (int): Maximum iterations for active learning
    - output_base_path_dict (dict): Mapping from GP name to output base path
    - gp_class_dict (dict): Mapping from GP name (str) to GP class
    - summary_results_dict (dict): Optional dict from GP name to list for results
    """
    
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    print(f"\n--- Processing: {dataset_name} ---")

    # --- Load and prepare data ---
    df = pd.read_csv(dataset_path)
    df[target_col] = np.log(df[target_col])
    features = [c for c in df.columns if c not in excluded_cols + [target_col]]
    target = [target_col]

    # --- Loop through GP variants ---
    for gp_name, gp_class in gp_class_dict.items():
        output_dir = os.path.join(output_base_path_dict[gp_name], dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n>> Running with GP: {gp_name}")

        # Run active learning loop
        mae_collection, mean_collection, stopping_index, gradient_stable_index = loop(
            df=df,
            features=features,
            target=target,
            init=init_indices,
            imax=max_iter,
            gp_class=gp_class
        )

        # Save MAE
        mae_df = pd.DataFrame({"Iteration": range(len(mae_collection)), "MAE": mae_collection})
        mae_df.to_csv(os.path.join(output_dir, f"mae_history_{gp_name}.csv"), index=False)

        # Save mean predictions per iteration
        mean_array = np.array([m.squeeze() for m in mean_collection])
        mean_df = pd.DataFrame(mean_array.T)
        mean_df.columns = [f"Iter_{i}" for i in range(mean_array.shape[0])]
        mean_df.insert(0, "Index", range(mean_array.shape[1]))
        mean_df.to_csv(os.path.join(output_dir, f"mean_predictions_per_iteration_{gp_name}.csv"), index=False)

        # Save final predictions
        final_pred = np.array(mean_collection[-1]).squeeze()
        final_true = np.exp(df[target_col].values).squeeze()
        final_df = pd.DataFrame({
            "True_Resistance": final_true,
            "Predicted_Resistance": final_pred
        })
        final_df.to_csv(os.path.join(output_dir, f"final_predictions_{gp_name}.csv"), index=False)

        # Save info
        with open(os.path.join(output_dir, f"info_{gp_name}.txt"), "w") as f:
            f.write(f"Stopped at iteration: {stopping_index}\n")
            f.write(f"Gradient stable index: {gradient_stable_index}\n")
            f.write(f"Init indices: {init_indices}\n")

        print(f"{gp_name} results saved to: {output_dir}")

        # Optional: Save summary
        if summary_results_dict is not None:
            if gp_name not in summary_results_dict:
                summary_results_dict[gp_name] = []
            summary_results_dict[gp_name].append({
                "Dataset": dataset_name,
                "Stopping Iteration": stopping_index,
                "Gradient Stable Index": gradient_stable_index,
                "Initial Indices": str(init_indices),
                "Total Iterations": len(mae_collection)
            })



def compare_evenly_vs_mixed(
    evenly_path,         # path to evenly spaced results file
    mixed_root,          # root folder containing mixed results
    mixed_filename,      # filename for mixed stopping indices
    output_dir,          # folder to save comparison results
    id_pattern=r"(\d{5})"  # regex pattern for extracting MaterialID
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load evenly spaced results ---
    evenly_df = pd.read_csv(evenly_path)
    evenly_df["MaterialID"] = evenly_df["Dataset"].str.extract(id_pattern)

    comparison_results = []
    top_strategies = []

    for _, row in evenly_df.iterrows():
        material_id = row["MaterialID"]
        evenly_stopping = int(row["Stopping Iteration"])

        mixed_file_path = os.path.join(mixed_root, f"{material_id}_results", mixed_filename)
        if not os.path.isfile(mixed_file_path):
            print(f"Skipping missing file for: {material_id}")
            continue

        mixed_df = pd.read_csv(mixed_file_path)

        # Filter only mixed strategies (contain '+')
        mixed_only = mixed_df[mixed_df["Strategy"].str.contains("\+")].copy()

        if not mixed_only.empty:
            # Normalize strategy names
            mixed_only["NormalizedStrategy"] = mixed_only["Strategy"].apply(
                lambda x: "+".join(sorted(x.split("+")))
            )

            # Count unique mixed strategies after normalization
            total_mixed = mixed_only["NormalizedStrategy"].nunique()

            # Group by normalized strategy and get all rows
            grouped = mixed_only.groupby("NormalizedStrategy")

            improved_strategies = []

            for _, group in grouped:
                # If any version of this mixed strategy has better performance
                better_than_evenly = group[group["StoppingIteration"] < evenly_stopping]
                if not better_than_evenly.empty:
                    improved_strategies.append(better_than_evenly)

            if improved_strategies:
                improved_df = pd.concat(improved_strategies)
                improved_count = improved_df["NormalizedStrategy"].nunique()
                improved_percent = (improved_count / total_mixed) * 100

                improved_df = improved_df.copy()
                improved_df["EvenlyStoppingIteration"] = evenly_stopping

                out_file = os.path.join(output_dir, f"{material_id}_mixed_better_than_evenly.csv")
                improved_df[["Strategy", "StoppingIteration", "EvenlyStoppingIteration"]].to_csv(out_file, index=False)

                # Select top improved strategy
                top_row = improved_df.loc[improved_df["StoppingIteration"].idxmin()]
                top_stopping = int(top_row["StoppingIteration"])
                top_strategy = top_row["Strategy"]
                improvement = evenly_stopping - top_stopping
                improvement_percent = round((improvement / evenly_stopping) * 100, 2)

                top_strategies.append({
                    "MaterialID": material_id,
                    "TopStrategy": top_strategy,
                    "StoppingIteration": top_stopping,
                    "EvenlyStoppingIteration": evenly_stopping,
                    "Improvement": improvement,
                    "ImprovementPercent": improvement_percent
                })
            else:
                improved_count = 0
                improved_percent = 0
        else:
            improved_count = 0
            total_mixed = 0
            improved_percent = 0

        comparison_results.append({
            "MaterialID": material_id,
            "EvenlyStoppingIteration": evenly_stopping,
            "ImprovedMixedCount": improved_count,
            "TotalMixedStrategies": total_mixed,
            "ImprovedPercent": round(improved_percent, 2)
        })

    # --- Save comparison summary ---
    summary_df = pd.DataFrame(comparison_results)
    summary_path = os.path.join(output_dir, "evenly_vs_mixed_comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    # --- Save top strategies and plot ---
    top_df = pd.DataFrame(top_strategies)
    if not top_df.empty:
        top_path = os.path.join(output_dir, "top_mixed_strategies.csv")
        top_df.to_csv(top_path, index=False)

        # --- Plot: Material ID on Y-axis, Stopping on X-axis ---
        sns.set_style("ticks")
        sns.set_context("talk", font_scale=1.2)
        plt.figure(figsize=(10, 6))

        y = range(len(top_df))
        colors = sns.color_palette("Set2", 2)

        # Plot horizontal bars
        plt.barh([i + 0.2 for i in y], top_df["EvenlyStoppingIteration"],
                 height=0.4, label="Four-point probe (Evenly)", color=colors[0])
        bars = plt.barh([i - 0.2 for i in y], top_df["StoppingIteration"],
                        height=0.4, label="Best Mixed", color=colors[1])

        # Annotate best strategies + % improvement
        for bar, strategy, percent in zip(bars, top_df["TopStrategy"], top_df["ImprovementPercent"]):
            width = bar.get_width()
            label = f"{strategy} ({percent}%)"
            plt.text(width + 1, bar.get_y() + bar.get_height() / 2,
                     label, va="center", fontsize=9)

        # Format axes
        plt.yticks(y, top_df["MaterialID"], fontsize=12)
        plt.xlabel("Iteration (# Measurements = Iteration + 9 or 10 Initial Points)", fontsize=16)
        plt.ylabel("Material Library ID", fontsize=16)
        #plt.title("Comparison of Four-point probe vs Best Mixed Strategies", fontsize=18, pad=20)
        plt.legend(fontsize=13, frameon=False)
        plt.tight_layout()

        # Save PDF
        combined_plot_path = os.path.join(output_dir, "comparison_all_materials_pub.pdf")
        plt.savefig(combined_plot_path, dpi=600, bbox_inches="tight", format="pdf")
        plt.close()