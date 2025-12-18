import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(pred, ref, title="Predicted vs Reference Energies", units="eV"):
    pred = np.array(pred)
    ref = np.array(ref)

    mae = mean_absolute_error(ref, pred)
    rmse = np.sqrt(mean_squared_error(ref, pred))
    r2 = r2_score(ref, pred)
    bias = np.mean(pred - ref)

    metrics_df = pd.DataFrame(
        {
            "MAE": [mae],
            "RMSE": [rmse],
            "R²": [r2],
            "Bias": [bias],
        }
    )

    return metrics_df


def plot_predictions(pred, ref, title="Predicted vs Reference Energies", units="mHa"):
    pred = np.array(pred)
    ref = np.array(ref)

    sorted_indices = np.argsort(ref)
    pred = pred[sorted_indices]
    ref = ref[sorted_indices]

    coeffs = np.polyfit(ref, pred, 1)
    fit_fn = np.poly1d(coeffs)

    plt.figure(figsize=(6, 6))
    plt.scatter(ref, pred, color="blue", label="Data")
    plt.plot(ref, ref, "k--", label="Ideal (y=x)")
    plt.plot(ref, fit_fn(ref), "r-", label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.3f}")
    plt.xlabel(f"Reference [{units}]")
    plt.ylabel(f"Predicted [{units}]")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    base_name = "data/"

    file_name_shortcuts = {
        "alcohols": "predictions_alcohols_esen.csv",
        "amino_acids": "predictions_amino_acids_esen.csv",
        "alkanes": "predictions_alkanes_esen.csv",
        "pubchem": "predictions_pubchem_esen.csv",
    }

    all_results = {}

    for key, file_name in file_name_shortcuts.items():
        path = base_name + file_name
        df = pd.read_csv(path)

        model_energies = df["ESEN_energy"].to_numpy()
        ref_energies = df["REF_energy"].to_numpy()

        all_results[key] = {
            "model_energies": model_energies,
            "ref_energies": ref_energies,
        }

    for key, result in all_results.items():
        metrics = evaluate_predictions(result["model_energies"], result["ref_energies"])
        print(key)
        print(metrics)

    for key, result in all_results.items():
        plot_predictions(
            result["model_energies"],
            result["ref_energies"],
            title=f"eSEN - {key}",
            units="mHa",
        )


if __name__ == "__main__":
    main()

