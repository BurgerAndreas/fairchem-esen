
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import matplotlib.pyplot as plt



def evaluate_predictions(pred, ref, title="Predicted vs Reference Energies", units="eV"):
    """
    Compute error metrics and plot predicted vs. reference values.

    Parameters
    ----------
    pred : array-like
        Predicted values (e.g., from MACE).
    ref : array-like
        Reference / true values.
    title : str
        Plot title.
    units : str
        Units to display on axis labels.
    """

    pred = np.array(pred)
    ref = np.array(ref)

    # Error metrics
    mae = mean_absolute_error(ref, pred)
    rmse = np.sqrt(mean_squared_error(ref, pred))
    r2 = r2_score(ref, pred)
    bias = np.mean(pred - ref)

    # Metrics table
    metrics_df = pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2],
        "Bias": [bias]
    })
    
    return metrics_df


def plot_predictions(pred, ref, title="Predicted vs Reference Energies", units="mHa"):
    """
    Plot predicted vs. reference values.

    Parameters
    ----------
    pred : array-like
        Predicted values (e.g., from MACE).
    ref : array-like
        Reference / true values.
    title : str
        Plot title.
    units : str
        Units to display on axis labels.
    """

    pred = np.array(pred)
    ref = np.array(ref)
    
    # sort by reference values
    sorted_indices = np.argsort(ref)
    pred = pred[sorted_indices]
    ref = ref[sorted_indices]
    

    # Linear regression fit (calibration)
    coeffs = np.polyfit(ref, pred, 1)
    fit_fn = np.poly1d(coeffs)

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(ref, pred, color='blue', label='Data')
    plt.plot(ref, ref, 'k--', label='Ideal (y=x)')
    plt.plot(ref, fit_fn(ref), 'r-', label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.3f}')
    plt.xlabel(f"Reference [{units}]")
    plt.ylabel(f"Predicted [{units}]")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()




# base_name = "data/validation_unseen/"
# file_names = ["predictions_alcohols.xyz","predictions_amino_acids.xyz","predictions_alkanes.xyz"]

base_name = "data/"

# file_name_shortcuts = {
#     "alcohols": "validation_unseen/predictions_alcohols.xyz",
#     "amino_acids": "validation_unseen/predictions_amino_acids.xyz",
#     "alkanes": "validation_unseen/predictions_alkanes.xyz",
#     "qm7_300_train": "predictions_8020_qm7_300_random_train.xyz",
#     "qm7_300_validation": "predictions_8020_qm7_300_random_validation.xyz",
# }

file_name_shortcuts = {
    "alcohols": "predictions_alcohols_mine.xyz",
    "amino_acids": "predictions_amino_acids_mine.xyz",
    "alkanes": "predictions_alkanes_mine.xyz",
    "qm7_300_train": "predictions_8020_qm7_300_random_train.xyz",
    "qm7_300_validation": "predictions_8020_qm7_300_random_validation.xyz",
    "pubchem_halfway": "predictions_pubchem_halfway_mine.xyz",
}


file_names = file_name_shortcuts.values()

all_results ={}
for file_name_shortcut, file_name in file_name_shortcuts.items():
    ref_energies = []
    mace_energies = []
    print("\n")
    print(file_name)
    
    with open(f"{base_name}/{file_name}", "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            num_atoms = int(line)
            props_line = lines[i+1]
            ref_match = re.search(r"REF_energy=([\-\d\.Ee]+)", props_line)
            mace_match = re.search(r"MACE_energy=([\-\d\.Ee]+)", props_line)
            try:
                ref_energy = float(ref_match.group(1)) if ref_match else None
                mace_energy = float(mace_match.group(1)) if mace_match else None
            except (ValueError, AttributeError):
                ref_energy = None
                mace_energy = None
            if ref_energy is not None and mace_energy is not None:
                ref_energies.append(ref_energy)
                mace_energies.append(mace_energy)
            i += num_atoms + 2
        else:
            i += 1
    
    # for idx, (ref, mace) in enumerate(zip(ref_energies, mace_energies)):
    #     print(f"Molecule {idx+1}: REF_energy={ref}, MACE_energy={mace}, Error={mace-ref}")

    all_results[file_name_shortcut] = {
        "mace_energies": mace_energies,
        "ref_energies": ref_energies,
    }




all_results


for key_word in all_results.keys():
    result_per_file=all_results[key_word]
    print(key_word)
    results = evaluate_predictions(result_per_file["mace_energies"], result_per_file["ref_energies"])
    print(results)



# all_results_pd = pd.DataFrame()
for key_word in all_results.keys():
    result_per_file=all_results[key_word]
    plot_predictions(
        result_per_file["mace_energies"], 
        result_per_file["ref_energies"],
        title=f"MACE - {key_word}", 
        units="mHa",
    )
    # break
    


import wandb

# need to construct: `entity/project/run_id`
wandb_entity = 'andreas-burger'
wandb_project = 'MD17'
wandb_runid = {
    "alkanes": "3l92yuc9",
    "alcohols": "yril2348",
    "amino_acids": "vahdkq5v",
    # "pubchem_halfway": "qwbew1r6",
    # "pubchem_halfway": "ddor5oct",
    "pubchem_halfway": "s9m1p9li",
}

keys = wandb_runid.keys()


run = wandb.Api().run(f'{wandb_entity}/{wandb_project}/{wandb_runid["alcohols"]}')
history = run.history()

ref_energies = - (history['test_mp2_energy_error_step'].to_numpy()[:-1])
model_energies_error = history['test_model_energy_error_step'].to_numpy()[:-1]
model_energies = ref_energies + model_energies_error



model_energies, ref_energies


for key in wandb_runid.keys():
    run = wandb.Api().run(f'{wandb_entity}/{wandb_project}/{wandb_runid[key]}')
    history = run.history()
    try:
        ref_energies = - (history['test_mp2_energy_error'].to_numpy()[:-1])
        model_energies_error = history['test_model_energy_error'].to_numpy()[:-1]
        model_energies = ref_energies + model_energies_error
    except KeyError:
        ref_energies = - (history['test_mp2_energy_error_step'].to_numpy()[:-1])
        model_energies_error = history['test_model_energy_error_step'].to_numpy()[:-1]
        model_energies = ref_energies + model_energies_error
    
    plot_predictions(
        model_energies, 
        ref_energies,
        title=f"LMBM - {key}", 
        units="mHa",
    )
    # break








