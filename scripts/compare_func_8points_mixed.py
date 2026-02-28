import os
import pandas as pd
from glob import glob

def compare_all_top5similarity_mix_with_top8(
    material_ids,
    base_root,
    mixed_folder,
    output_file="comparison_top5_mixed_vs_top8.csv",
    top8_name="Top8Similarity",
    prefix="Top5Similarity",
    total_iterations=100  # <-- New parameter
):
    all_results = []
    improved_count = 0
    total_valid_comparisons = 0  # Excludes "Top8 Not Found"

    for material_id in material_ids:
        base_file = os.path.join(base_root, f"{material_id}_results", "mae_priors_stopping_indices.csv")
        mixed_file = os.path.join(mixed_folder, f"{material_id}_results", "mae_priors_stopping_indices.csv")

        if not os.path.isfile(base_file) or not os.path.isfile(mixed_file):
            print(f"Missing file for {material_id}, skipping.")
            continue

        base_df = pd.read_csv(base_file)
        mixed_df = pd.read_csv(mixed_file)

        # Get Top8Similarity stopping iteration
        top8_row = base_df[base_df["Strategy"] == top8_name]
        if top8_row.empty:
            print(f"{top8_name} not found in {material_id}")
            top8_stop = None
        else:
            top8_stop = int(top8_row["StoppingIteration"].values[0])

        # Filter only Top5Similarity+ mixed strategies
        top5_mixed_df = mixed_df[mixed_df["Strategy"].str.startswith(f"{prefix}+")]

        for _, row in top5_mixed_df.iterrows():
            mixed_strategy = row["Strategy"]
            mixed_stop = int(row["StoppingIteration"])

            if top8_stop is None:
                result = "Top8 Not Found"
                improvement_percentage = None
            elif mixed_stop < top8_stop:
                result = "Improved"
                improvement_percentage = round(100 * (top8_stop - mixed_stop) / total_iterations, 2)
                improved_count += 1
                total_valid_comparisons += 1
            elif mixed_stop > top8_stop:
                result = "Worse"
                improvement_percentage = round(100 * (top8_stop - mixed_stop) / total_iterations, 2)
                total_valid_comparisons += 1
            else:
                continue  # Skip Equal

            all_results.append({
                "MaterialLibrary": material_id,
                "MixedStrategy": mixed_strategy,
                "MixedStoppingIteration": mixed_stop,
                "Top8StoppingIteration": top8_stop,
                "ComparisonResult": result,
                "ImprovementPercentage": improvement_percentage
            })

    # Save detailed results
    df = pd.DataFrame(all_results)


    # Calculate overall % improvement
    if total_valid_comparisons > 0:
        overall_improvement_percent = round(100 * improved_count / total_valid_comparisons, 2)
        print(f"Overall Improvement Rate: {overall_improvement_percent}% ({improved_count}/{total_valid_comparisons})")
    else:
        overall_improvement_percent = None
        print("No valid comparisons found (Top8 missing or all were equal).")

    return df, overall_improvement_percent




# --- helper: normalize pair names (so "A+B" and "B+A" are treated the same) ---
def _canonical_pair(mixed_strategy: str) -> str:
    a, b = mixed_strategy.split("+", 1)
    return "+".join(sorted([a.strip(), b.strip()], key=str.lower))


# --- 1. Compare fixed base strategies with mixed strategies ---
def compare_fixed_base_mixed_strategies(
    material_ids,
    base_root,
    mixed_folder,
    base_strategies,
    mixing_strategy,
    output_file="comparison_fixed_base_mix_vs_original.csv",
    total_iterations=100,
):
    all_results = []
    improved_count = 0
    total_valid_comparisons = 0

    for material_id in material_ids:
        base_file = os.path.join(base_root, f"{material_id}_results", "mae_priors_stopping_indices.csv")
        mixed_file = os.path.join(mixed_folder, f"{material_id}_results", "mae_priors_stopping_indices.csv")

        if not os.path.isfile(base_file) or not os.path.isfile(mixed_file):
            print(f"Missing file for {material_id}, skipping.")
            continue

        base_df = pd.read_csv(base_file)
        mixed_df = pd.read_csv(mixed_file)

        # check required columns
        if not {"Strategy", "StoppingIteration"}.issubset(base_df.columns) or not {"Strategy", "StoppingIteration"}.issubset(mixed_df.columns):
            print(f"Missing columns in data for {material_id}, skipping.")
            continue

        for strategy in base_strategies:
            if strategy == mixing_strategy:
                continue  # skip comparing mixing strategy with itself

            mix_name_1 = f"{mixing_strategy}+{strategy}"
            mix_name_2 = f"{strategy}+{mixing_strategy}"

            mixed_row = mixed_df[mixed_df["Strategy"].isin([mix_name_1, mix_name_2])]
            base_row = base_df[base_df["Strategy"] == strategy]

            if mixed_row.empty or base_row.empty:
                continue

            # choose the best mixed row (smallest stopping iteration)
            best_mixed = mixed_row.sort_values("StoppingIteration").iloc[0]
            mixed_strategy_name = str(best_mixed["Strategy"])
            mixed_stop = int(best_mixed["StoppingIteration"])
            base_stop = int(base_row["StoppingIteration"].values[0])

            if mixed_stop == base_stop:
                continue

            result = "Improved" if mixed_stop < base_stop else "Worse"
            improvement_percentage = round(100 * (base_stop - mixed_stop) / total_iterations, 2)

            if result == "Improved":
                improved_count += 1
            total_valid_comparisons += 1

            all_results.append({
                "MaterialLibrary": material_id,
                "BaseStrategy": strategy,
                "MixedWith": mixing_strategy,
                "MixedStrategy": mixed_strategy_name,
                "CanonicalMixedStrategy": _canonical_pair(mixed_strategy_name),
                "BaseStoppingIteration": base_stop,
                "MixedStoppingIteration": mixed_stop,
                "ComparisonResult": result,
                "ImprovementPercentage": improvement_percentage
            })

    # save detailed results
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    # print improvement rate
    if total_valid_comparisons > 0:
        overall_improvement_percent = round(100 * improved_count / total_valid_comparisons, 2)
        print(f"\nOverall Improvement Rate for {mixing_strategy}: {overall_improvement_percent}% "
              f"({improved_count}/{total_valid_comparisons})")
    else:
        overall_improvement_percent = None
        print(f"No valid comparisons found for {mixing_strategy}.")

    # make per-run summary
    if not df.empty:
        improved_df = df[df["ComparisonResult"] == "Improved"].copy()
        improved_df = improved_df.drop_duplicates(
            subset=["MaterialLibrary", "BaseStrategy", "CanonicalMixedStrategy"]
        )
        summary = (improved_df
                   .groupby("CanonicalMixedStrategy")
                   .size()
                   .reset_index(name="ImprovedCount")
                   .sort_values("ImprovedCount", ascending=False))
        summary_file = output_file.replace(".csv", "_summary.csv")
        summary.to_csv(summary_file, index=False)
    else:
        summary = pd.DataFrame()

    return df, overall_improvement_percent, summary
import os
import pandas as pd
from glob import glob

def merge_all_summaries(input_folder, pattern="*_mixed_vs_others_summary.csv",
                        output_file="combined_mixed_strategy_summary.csv"):
    files = glob(os.path.join(input_folder, pattern))
    if not files:
        print("No summary files found.")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if "CanonicalMixedStrategy" in d.columns and "ImprovedCount" in d.columns:
                frames.append(d[["CanonicalMixedStrategy", "ImprovedCount"]])
            else:
                print(f"Skipping {f}: unexpected columns {list(d.columns)}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not frames:
        print("No valid summary data found.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # 🔑 Keep maximum ImprovedCount for each CanonicalMixedStrategy
    final_summary = (
        combined
        .groupby("CanonicalMixedStrategy")["ImprovedCount"]
        .max()
        .reset_index()
        .rename(columns={"ImprovedCount": "MaxImprovedCount"})
        .sort_values(by="MaxImprovedCount", ascending=False)
    )

    output_path = os.path.join(input_folder, output_file)
    final_summary.to_csv(output_path, index=False)
    #print(f"Saved merged summary to {output_path}")
    return final_summary

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob

def generate_mixed_strategy_heatmap(data_folder, pattern="*_mixed_vs_others.csv",
                                     stopping_iteration=232, top_k=30):
    all_data = []

    for file in glob(os.path.join(data_folder, pattern)):
        df = pd.read_csv(file)
        if not {"MaterialLibrary", "CanonicalMixedStrategy", "ImprovementPercentage"}.issubset(df.columns):
            print(f"Skipping {file}, missing columns.")
            continue
        all_data.append(df)

    if not all_data:
        print("No valid input data found.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # ✅ Only keep positive improvements
    combined_df = combined_df[combined_df["ImprovementPercentage"] > 0]

    if combined_df.empty:
        print("No positive improvements found.")
        return

    # ✅ Replace with Improvement + stopping iteration (default 242)
    combined_df["MixedStopValue"] = combined_df["ImprovementPercentage"] + stopping_iteration

    # --- Find top-k strategies by total MixedStopValue ---
    top_strategies = (
        combined_df.groupby("CanonicalMixedStrategy")["MixedStopValue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    # Filter to only include top-k strategies
    filtered_df = combined_df[combined_df["CanonicalMixedStrategy"].isin(top_strategies)]

    # Pivot to create heatmap matrix (now using MixedStopValue)
    heatmap_data = filtered_df.pivot_table(
        index="MaterialLibrary",
        columns="CanonicalMixedStrategy",
        values="MixedStopValue",
        aggfunc="max"  # Use max if duplicates exist
    )

    # --- Plot heatmap ---
    plt.figure(figsize=(1.5 * top_k, 8))
    sns.set(font_scale=0.9)
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar_kws={'label': 'Total Measurement Reduction'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f"Top {top_k} Mixed Strategies (Stopping Iteration + {stopping_iteration})")
    plt.ylabel("Materials Library ID")
    plt.xlabel("Canonical Mixed Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
