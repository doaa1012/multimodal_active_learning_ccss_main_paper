import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import os
def generate_random_initializations_to_json(
    X,
    seeds,              # <-- ADD THIS
    n_init=5,
    output_path="random_inits.json"
):
    random_inits = {}

    for seed in seeds:   # <-- NOT range(n_runs) anymore
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=n_init, replace=False).tolist()
        random_inits[f"Random_seed_{seed}"] = indices

    with open(output_path, "w") as f:
        json.dump(random_inits, f, indent=4)

def generate_random_inits_for_all_datasets(
    datasets,
    output_dir,
    seeds,  # <-- pass here too
    target_col="Resistance",
    excluded_cols=["ID", "x", "y"],
    n_init=5,
):
    os.makedirs(output_dir, exist_ok=True)

    for dataset_path in datasets:
        df = pd.read_csv(dataset_path)

        filename = os.path.basename(dataset_path)
        dataset_id = filename.split("_")[0]

        features = [col for col in df.columns if col not in excluded_cols + [target_col]]
        X = df[features].values

        json_path = os.path.join(output_dir, f"{dataset_id}_indices.json")

        generate_random_initializations_to_json(
            X,
            seeds=seeds,
            n_init=n_init,
            output_path=json_path
        )



def load_mae_dict_from_results(results_folder):
    pkl_path = os.path.join(results_folder, "mae_priors_all_results.pkl")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data["mae"]  

def mae_dict_to_matrix(mae_dict):
    curves = list(mae_dict.values())
    max_len = max(len(c) for c in curves)

    mat = np.full((len(curves), max_len), np.nan)

    for i, c in enumerate(curves):
        mat[i, :len(c)] = c

    return mat


def plot_mae_seeds(mae_matrix, title="Random Initialization (30 seeds)"):

    mean_mae = np.nanmean(mae_matrix, axis=0)
    std_mae  = np.nanstd(mae_matrix, axis=0)

    # faint individual runs
    for run in mae_matrix:
        plt.plot(run, color='gray', alpha=0.15)

    # bold mean curve
    plt.plot(mean_mae, color='black', linewidth=3, label='Mean MAE')

    # variability band
    plt.fill_between(
        range(len(mean_mae)),
        mean_mae - std_mae,
        mean_mae + std_mae,
        color='black',
        alpha=0.2
    )

    plt.xlabel("Active Learning Iteration")
    plt.ylabel("MAE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
