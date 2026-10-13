# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
import scipy.optimize as opti
import ODRBootstrapping_edits_Fall_2024 as boot
import janitor
from matplotlib.backends.backend_pdf import PdfPages
import traceback
import math
from matplotlib import rc
# %%
excel_path = "SIMS Calibrations Caltech 2020 April 2025 Update.xlsx"

Calibrations = pd.read_excel(
    excel_path,
    sheet_name="Combined Calibrations",
    # header=1,
    # index='Unique_ID',
    engine="openpyxl",
)

# pyjanitor cleaning of dataset
Calibrations = Calibrations.remove_empty()  # .dropna()

Calibrations.set_index("File", inplace=True)

#%%
element_data = {
    "H2O": {
        "counts": "17O/30Si",
        "SiO2": "SiO2",
        "concentration": "H2O ppm",
        "concentration_sigma": "H2O_ppm_sigma",
        "counts_sigma": "17O/30Si_Sigma",
    },
    "CO2": {
        "counts": "12C/30Si",
        "SiO2": "SiO2",
        "concentration": "CO2 ppm",
    },  # Add sigma if available
    "S": {"counts": "32S/30Si", "SiO2": "SiO2", "concentration": "S ppm"},  # Add sigma if available
    "Cl": {
        "counts": "35Cl/30Si",
        "SiO2": "SiO2",
        "concentration": "Cl ppm",
    },  # Add sigma if available
    "F": {"counts": "19F/30Si", "SiO2": "SiO2", "concentration": "F ppm"},  # Add sigma if available
}
# %%
# %%
def line_func(x, slope, intercept):
    """
    x: array of x points
    """
    return slope * x + intercept


def Linear_R2(X, Y, slope, intercept=0):
    Predicted = X * slope + intercept

    residuals = Y - Predicted
    mean_subtracted = Y - Y.values.mean()

    R2 = 1 - np.sum(residuals**2) / np.sum(mean_subtracted**2)
    return R2


def weighted_fit(x, y, p0, sigma):
    popt, pcov = opti.curve_fit(line_func, x, y, p0, sigma=sigma)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

#%%

for calibrations