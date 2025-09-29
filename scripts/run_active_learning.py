import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import pickle
import traceback
from pyDOE2 import lhs
from scripts.measurement_devices import Resistance
from scripts.gaussian_process_basic import GPBasic
from scripts.gaussian_process_sawei import GPSawei
import json

def loop(df, features, target, init, imax, gp_class=GPBasic, mae_threshold=0.005):
    device = Resistance(df, features=features, target=target)
    X = device.get_features()
    y_ref = device.df[target[0]]

    # Initial measurements
    X0, y0 = device.get_initial_measurement(indices=init, target_property=target[0])
    model = gp_class(X0, y0, device.features)

    # First prediction
    model.predict(X)
    resistance1 = model.mu
    error = MAE(np.exp(resistance1), np.exp(y_ref))

    # Initialize collections
    max_cov, index_max_cov = model.get_max_covariance()
    mae_collection = [error]
    mean_collection = [np.exp(resistance1)]
    stopping_index = imax
    gradient_stable_index = None
    # Active learning iterations
    for i in range(imax):
        # Get new measurement
        X_tmp, y_tmp = device.get_measurement(
            indices=[index_max_cov], target_property=target[0]
        )
        model.update_Xy(X_tmp, y_tmp)
        model.predict(X)

        # Calculate error and save results
        error = MAE(np.exp(model.mu), np.exp(y_ref))
        mae_collection.append(error)
        mean_collection.append(np.exp(model.mu))

        # Print status
        print(f"Iteration {i+1}: MAE = {error:.5f}")

        stop, gradient_idx = model.check_stopping_condition(X, np.exp(y_ref), mae_threshold=mae_threshold)

        # Store the first time the gradient becomes stable
        if gradient_stable_index is None and gradient_idx is not None:
            gradient_stable_index = gradient_idx

        if stop:
            stopping_index = i + 1
            print(f"Stopping early at iteration {stopping_index} due to stopping condition.")
            break

        # Choose next point
        max_cov, index_max_cov = model.get_max_covariance()

    del device
    del model

    return mae_collection, mean_collection, stopping_index, gradient_stable_index



def select_initial_indices(X, n_init=5, seed=42):
    rng = np.random.default_rng(seed)  

    selection_results = {
        "Random": rng.choice(len(X), size=n_init, replace=False).tolist(),
        "LHS": [],
        "K-Means": [],
        "Farthest": [],
        "ODAL": [],
        "K-Center": []
    }

    # LHS (Latin Hypercube Sampling)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    lhs_points = lhs(X.shape[1], samples=n_init, criterion='center')  # deterministic given seed
    distances = np.linalg.norm(X_scaled[:, None] - lhs_points[None], axis=2)
    selection_results["LHS"] = np.argmin(distances, axis=0).tolist()

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_init, random_state=seed).fit(X)
    _, indices = np.unique(kmeans.labels_, return_index=True)
    selection_results["K-Means"] = indices.tolist()

    # Farthest Point Sampling
    selected = [rng.integers(len(X))]
    for _ in range(n_init - 1):
        dists = np.min(np.linalg.norm(X - X[selected][:, None], axis=2), axis=0)
        next_idx = np.argmax(dists)
        selected.append(next_idx)
    selection_results["Farthest"] = selected

    # ODAL
    iso = IsolationForest(contamination=0.05, random_state=seed)
    scores = iso.fit_predict(X)
    selection_results["ODAL"] = np.where(scores == -1)[0][:n_init].tolist()

    # K-Center Greedy
    selected_kc = [rng.integers(len(X))]
    while len(selected_kc) < n_init:
        dists = np.min(np.linalg.norm(X - X[selected_kc][:, None], axis=2), axis=0)
        next_idx = np.argmax(dists)
        selected_kc.append(next_idx)
    selection_results["K-Center"] = selected_kc

    return selection_results


def generate_full_merged_strategies(init_strategies):
    """
    Generate merged versions of initial active learning strategies using
    full merge approach (union of all points), including all ordered pairs.

    Args:
        init_strategies (dict): Dictionary of {strategy_name: [indices]}.

    Returns:
        dict: A dictionary of all strategies including original and merged ones.
    """
    merged_strategies = {}

    strategy_names = list(init_strategies.keys())
    for i in range(len(strategy_names)):
        for j in range(len(strategy_names)):
            if i != j:
                name1 = strategy_names[i]
                name2 = strategy_names[j]
                merged_name = f"{name1}+{name2}"
                merged_strategies[merged_name] = list(set(init_strategies[name1]) | set(init_strategies[name2]))

    # Combine original and merged
    all_strategies = {**init_strategies, **merged_strategies}
    return all_strategies



def plot_final_predictions_indexed(y_true, y_pred, dataset_name, output_path=None):
    """
    Plot indexed predictions and true values with color mapping based on predictions.

    Args:
        y_true (array): True values (exp-scaled).
        y_pred (array): Predicted values (exp-scaled).
        dataset_name (str): Name for title and file naming.
        output_path (str, optional): If given, saves the plot instead of displaying it.
    """
    indices = np.arange(len(y_true))
    plt.figure(figsize=(10, 6))

    # Plot true values as black markers
    plt.plot(indices, y_true, 'o', label='True Resistance', color='black', markersize=4)

    # Predicted values colored by prediction value
    scatter = plt.scatter(indices, y_pred, c=y_pred, cmap='viridis', label='Predicted Resistance',
                          s=40, edgecolor='k', alpha=0.9)

    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Resistance (Ω)", fontsize=12)
    plt.title(f"{dataset_name} – Indexed Prediction", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.colorbar(scatter, label="Predicted Resistance (Ω)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
       
    else:
        plt.show()

# Identity function: returns the original dictionary unchanged
def no_merge(init_dict):
    return init_dict

def run_active_learning_experiment(
    datasets,
    init_json_dir,
    output_base_path,
    generate_full_merged_strategies,
    loop_function,
    ResistanceClass,
    GPModelClass,
    #plot_final_predictions_indexed_func,
    target_col="Resistance",
    excluded_cols=["ID", "x", "y"]
):
    for dataset_path in datasets:
        try:
            dataset_name = os.path.basename(dataset_path).split("_")[0]  
            print(f"\nProcessing dataset: {dataset_name}")
           
            data_exp = pd.read_csv(dataset_path)

            if data_exp.empty:
                print(f"Warning: Dataset {dataset_name} is empty.")
                continue

            json_path = os.path.join(init_json_dir, f"{dataset_name}_indices.json")
            

            if not os.path.exists(json_path):
                print(f"JSON file not found for {dataset_name}, skipping.")
                continue

            with open(json_path, "r") as f:
                init_choices = json.load(f)

            init_choices = generate_full_merged_strategies(init_choices)
            print(init_choices)

            all_columns = data_exp.columns.tolist()
            features = [col for col in all_columns if col not in excluded_cols + [target_col]]
            target = [target_col]

            data_exp[target] = np.log(data_exp[target])

            output_dir = os.path.join(output_base_path, dataset_name + "_results")
            os.makedirs(output_dir, exist_ok=True)

            device = ResistanceClass(data_exp, features=features, target=target)

            params_exp = {
                "df": data_exp,
                "features": features,
                "target": target,
                "max_iter": 100,
            }

            mae_priors = {}
            mean_priors = {}
            stopping_indices = {}
            gradient_indices = {}

            for prior in init_choices:
                print(f"\nRunning strategy: {prior}")
                try:
                    init = init_choices[prior]
                    mae_tmp, mean_tmp, stop_idx, grad_idx = loop_function(
                        params_exp["df"], params_exp["features"], params_exp["target"], init, params_exp["max_iter"], gp_class=GPModelClass
                    )

                    mae_priors[prior] = mae_tmp
                    mean_priors[prior] = mean_tmp
                    stopping_indices[prior] = stop_idx
                    gradient_indices[prior] = grad_idx

                    final_pred = mean_tmp[-1]
                    final_true = np.exp(data_exp[target_col].values)

                    pred_df = pd.DataFrame({
                        "True Resistance": final_true.flatten(),
                        "Predicted Resistance": final_pred.flatten()
                    })

                    pred_csv_path = os.path.join(output_dir, f"{prior}_final_predictions.csv")
                    pred_df.to_csv(pred_csv_path, index=False)

                    #plot_final_predictions_indexed_func(
                    #    y_true=final_true,
                    #    y_pred=final_pred,
                    #    dataset_name=f"{dataset_name}_{prior}",
                    #    output_path=os.path.join(output_dir, f"{prior}_indexed_predictions_plot.png")
                    #)

                except Exception as strategy_error:
                    print(f"Error in strategy '{prior}' for dataset {dataset_name}: {strategy_error}")
                    traceback.print_exc()

            # Save summary results
            max_len = max(len(v) for v in mae_priors.values())
            mae_df = pd.DataFrame({k: v + [None] * (max_len - len(v)) for k, v in mae_priors.items()})
            mae_df.to_csv(os.path.join(output_dir, "mae_priors_results.csv"), index=False)

            stopping_df = pd.DataFrame({
                "StoppingIteration": pd.Series(stopping_indices),
                "GradientStableIteration": pd.Series(gradient_indices)
            })
            stopping_df.index.name = "Strategy"
            stopping_df.to_csv(os.path.join(output_dir, "mae_priors_stopping_indices.csv"))

            with open(os.path.join(output_dir, "mae_priors_all_results.pkl"), "wb") as f:
                pickle.dump({
                    "mae": mae_priors,
                    "stop_idx": stopping_indices,
                    "grad_idx": gradient_indices
                }, f)

            print(f"Finished processing dataset: {dataset_name}")

        except Exception as e:
            print(f"\nError processing dataset {dataset_path}: {e}")
            traceback.print_exc()


