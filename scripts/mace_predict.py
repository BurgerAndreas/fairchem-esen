
# !mace_eval_configs --configs data/alcohols.xyz --model qm7_300_random_train_60_40_split.model --output data/predictions_alcohols_60_40.xyz


# !mace_eval_configs --configs data/alkanes.xyz --model qm7_300_random_train_80_20_split.model --output data/predictions_alkanes.xyz


# !mace_eval_configs --configs data/amino_acids.xyz --model qm7_300_random_train_80_20_split.model --output data/predictions_amino_acids.xyz


# !mace_calculate_errors --configs data/predictions_alcohols.xyz --energy_key energy --ref_energy_key REF_energy --output data/predictions_errors_set_alcohols.txt



import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_name = "/home/mghazi/scratch/mace/data/validation_unseen/"
file_names = ["predictions_alcohols.xyz","predictions_amino_acids.xyz","predictions_alkanes.xyz"]


all_results ={}
for file_name in file_names:
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
    
    for idx, (ref, mace) in enumerate(zip(ref_energies, mace_energies)):
        print(f"Molecule {idx+1}: REF_energy={ref}, MACE_energy={mace}, Error={mace-ref}")
        
    all_results[file_name.split(".")[0]] ={
                                             "mace_energies":mace_energies,
                                             "ref_energies":ref_energies,
                                          }





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

    # Linear regression fit (calibration)
    coeffs = np.polyfit(pred, ref, 1)
    fit_fn = np.poly1d(coeffs)

    # Plot
    # plt.figure(figsize=(6,6))
    # plt.scatter(ref, pred, color='blue', label='Data')
    # plt.plot(ref, ref, 'k--', label='Ideal (y=x)')
    # plt.plot(ref, fit_fn(pred), 'r-', label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.3f}')
    # plt.xlabel(f"Reference [{units}]")
    # plt.ylabel(f"Predicted [{units}]")
    # plt.legend()
    # plt.title(title)
    # plt.grid(True)
    # plt.show()

    # Metrics table
    metrics_df = pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2],
        "Bias": [bias]
    })
    
    return metrics_df


for key_word in ["predictions_alkanes","predictions_amino_acids","predictions_alcohols"]:
    result_per_file=all_results[key_word]
    print(key_word)
    results = evaluate_predictions(result_per_file["mace_energies"], result_per_file["ref_energies"])
    print(results)





base_name = "/home/mghazi/scratch/mace/data/validation_unseen/"
file_names = ["predictions_alcohols.xyz","predictions_amino_acids.xyz","predictions_alkanes.xyz"]


all_results ={}
for file_name in file_names:
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
    
    for idx, (ref, mace) in enumerate(zip(ref_energies, mace_energies)):
        print(f"Molecule {idx+1}: REF_energy={ref}, MACE_energy={mace}, Error={mace-ref}")
        
    all_results[file_name.split(".")[0]] ={
                                             "mace_energies":mace_energies,
                                             "ref_energies":ref_energies,
                                          }




