import yaml
import os
import logging
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

# Assuming main.py is in the same directory and AnomalyDetection is importable
from main import AnomalyDetection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("grid_search.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def run_experiment(dataset, combo_id, params):
    """
    Run a single experiment for a given parameter combination.
    This function is designed to be run in parallel.
    """
    try:
        with open("config.yaml", "r") as f:
            base_config = yaml.safe_load(f)

        config = base_config.copy()
        config["model"]["gamma"] = params["gamma"]
        config["model"]["sigma"] = params["sigma"]
        config["model"]["lambda_smooth"] = params["lambda_smooth"]
        config["model"][dataset]["e_layers"] = params["e_layers"]
        config["training"]["lr"] = params["lr"]
        config["training"]["k"] = params["k"]
        config["testing"]["temperature"] = params["temperature"]

        # Create a unique temp config path to avoid conflicts in parallel runs
        temp_config_path = f"temp_config_{dataset}_{combo_id}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        # Initialize and run the detector
        detector = AnomalyDetection(temp_config_path, dataset)
        detector.train()
        accuracy, precision, recall, f_score = detector.test()

        # Clean up temp file
        os.remove(temp_config_path)

        return params, f_score, accuracy, precision, recall
    except Exception as e:
        logger.error(f"Error in combo {combo_id} with params {params}: {e}")
        return params, float("-inf"), 0, 0, 0  # Mark as failed with low score

def grid_search(dataset="SMAP", max_workers=None):
    """
    Perform efficient grid search using multiprocessing for parallel execution.
    Adjusts number of workers based on CPU cores, but limits to avoid GPU contention.
    Since the model uses MPS (Apple Silicon GPU), we limit parallelism to avoid resource conflicts.
    """
    logger.info(f"Starting grid search for dataset: {dataset}")

    param_grid = {
        "gamma": [1.0, 2.0, 3.0],
        "sigma": [0.5, 1.0, 1.5],
        "lambda_smooth": [0.001, 0.01, 0.1],
        "e_layers": [2, 3, 4],
        "lr": [1e-6, 1e-5, 1e-4],
        "k": [1, 3, 5],
        "temperature": [10, 50, 100],
    }

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    logger.info(f"Total combinations: {len(combinations)}")

    best_score = float("-inf")
    best_params = None
    results = []

    # Use multiprocessing Pool for parallel execution
    # Limit workers to min(4, cpu_count()) to avoid overwhelming MPS device
    if max_workers is None:
        max_workers = min(4, cpu_count())  # Conservative for GPU usage

    with Pool(processes=max_workers) as pool:
        # Partial function to fix dataset and pass combo_id
        func = partial(run_experiment, dataset)
        # Map over combinations with unique IDs
        async_results = [
            pool.apply_async(func, args=(i, dict(zip(keys, combo))))
            for i, combo in enumerate(combinations)
        ]

        for res in async_results:
            params, f_score, accuracy, precision, recall = res.get()
            results.append(
                {
                    "params": params,
                    "f_score": f_score,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                }
            )

            if f_score > best_score:
                best_score = f_score
                best_params = params

    logger.info("Grid search completed.")
    logger.info(f"Best params: {best_params}")
    logger.info(f"Best F-score: {best_score}")

    # Save results to YAML
    with open(f"grid_search_results_{dataset}.yaml", "w") as f:
        yaml.dump(results, f)

    return best_params, best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search for anomaly detection.")
    parser.add_argument("--dataset", type=str, default="SMAP", help="Dataset to use (e.g., SMAP or MSL)")
    args = parser.parse_args()
    
    grid_search(dataset=args.dataset)
