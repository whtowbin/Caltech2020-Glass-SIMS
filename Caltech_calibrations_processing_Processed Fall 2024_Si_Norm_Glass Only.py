# %%
from inspect import FullArgSpec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
import scipy.optimize as opti
import ODRBootstrapping_edits_May_2022 as boot
import janitor
from matplotlib.backends.backend_pdf import PdfPages
import traceback
import math
from matplotlib import rc

# %%
excel_path = "NS_Pyroxene_Glass-calib_and Data_Caltech2020_Processed2024.xlsx"
# "/Users/henry/Documents/Research/Writing/SIMS Olivine Submission Docs/Olivine_6F_calibrations_cleaned.xlsx"

Calibrations = pd.read_excel(
    excel_path,
    sheet_name="raw_data_calib",
    # header=1,
    # index='Unique_ID',
    engine="openpyxl",
)

# pyjanitor cleaning of dataset
Calibrations = Calibrations.remove_empty()

Calibrations.set_index("File", inplace=True)
# Calibrations.index = Calibrations["Unique_ID"]


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


# def calculate_values(index, row, calibration):
#     # Calculte Median Point and 95% confidence interval
#     Values = (
#         row["16OH/30Si"] * row["SiO2 (wt%)"] * calibration[:, 0] + calibration[:, 1]
#     )
#     bestfit = Values[0]
#     median = np.quantile(Values, 0.5)
#     N_err = np.quantile(Values, 0.05) - np.quantile(Values, 0.5)
#     P_err = np.quantile(Values, 0.95) - np.quantile(Values, 0.5)
#     return [index, bestfit, median, N_err, P_err]


# %%
# Plot Calinbration Lines for each phase and save as a pdf
Phases = Calibrations.Phase.unique()
Calculated = {}
Calibration_outputs = {}
day = np.datetime64("2020-02")

pp = PdfPages("Caltech 2020 NS Calibrations_30Si_Norm.pdf")


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.3), dpi=300)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
axs = axs.flatten()
for idx, phase in enumerate(["glass", "opx", "olivine", "cpx"]):  # Phases[2:3]:
    if phase not in list(Calculated):
        Calculated[phase] = []

    try:
        # fig, ax = plt.subplots(figsize=(6,6) ,dpi=200)
        Session = Calibrations.loc[
            # (Calibrations["Phase"] == phase)
            ((Calibrations["Phase"] == phase) | (Calibrations["Phase"] == "blank"))
        ]
        if phase not in list(Session.Phase.unique()):
            continue

        X, Y, sigma_y, sigma_x = (
            Session["17O/30Si"] * Session["SiO2"],
            Session["H2O ppm"],
            Session["H2O_ppm_sigma"],
            Session["17O/30Si_Sigma"],
        )

        # plt.plot(X, Y, linestyle="none", marker="o")
        axs[idx].errorbar(
            X,
            Y,
            yerr=sigma_y,
            xerr=sigma_x,
            linestyle="none",
            marker="o",
            ecolor="k",
            capsize=2,
            markeredgecolor="k",
        )

        fit1 = stats.linregress(X, Y)
        # print("slope: " + str(fit1.slope) + ", intercept: " + str(fit1.intercept))

        popt, perr = weighted_fit(X, Y, (fit1.slope, fit1.intercept), sigma_y)
        # print(popt)
        # print(perr)

        LineMax = X.max() * 1.05
        LineInterval = LineMax / 200

        output = boot.ODR_Bootstrap(
            x=X,
            y=Y,
            y_err=sigma_y,
            x_err=sigma_x,
            resample_draws=5000,
            LineMax=LineMax,
            LineInterval=LineInterval,
            InterceptFit=True,
            InitialGuess=[popt[0], popt[1]],
            Confidence_Bound=0.95,
        )
        boot.plot_regression(
            output[0],
            ax=axs[idx],
            LineMax=LineMax,
            LineInt=LineInterval,
        )
        # print(f"odr: {output[1]}")
        # Calibration_log[np.datetime_as_string(day, "M")] = output
        axs[idx].set_xlim(left=0)
        axs[idx].set_ylim(bottom=-10)
        axs[idx].set_ylabel("H2O (ppm)")
        axs[idx].set_xlabel(r"$\frac{^{16}OH}{^{30}Si}$ * SiO2 wt%")
        axs[idx].set_title(np.datetime_as_string(day, "M") + " " + phase)

        # Cal_date_DF = Calibrations.loc[(Calibrations["Date"] == day)]
        # for index, row in Cal_date_DF.iterrows():
        #     Calculated[phase].append(
        #         calculate_values(index, row, np.array(output[3]))
        #     )

        slopes = np.array(output[3])[:, 0]
        intercepts = np.array(output[3])[:, 1]
        Calibration_outputs[(np.datetime_as_string(day, "M"), phase)] = {
            "slopes": slopes,
            "intercepts": intercepts,
        }

        axs[idx].annotate(
            f"Y = {round(output[1][0])}X(±{round(np.std(slopes))}) + {round(output[1][1])}(±{round(np.std(intercepts))})",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
        )
        axs[idx].annotate(
            rf"$R^{2}$ = {round(Linear_R2(X,Y,output[1][0],output[1][1]),3)}",
            xy=(0.05, 0.90),
            xycoords="axes fraction",
        )

    except Exception as e:
        print("An Exception Occurred: ")
        traceback.print_exc()
        print(phase)
        print(day)
        print(e)

plt.show()
pp.savefig(fig)
pp.close()


# %%
def Linear_R2(points, slope, intercept=0):
    Predicted = points["x"].values * slope + intercept

    residuals = points["y"].values - Predicted
    mean_subtracted = points["y"].values - points["y"].values.mean()

    R2 = 1 - np.sum(residuals**2) / np.sum(mean_subtracted**2)
    return R2


# %%


#
def calculate_values(index, row, calibration):
    # Calculate Median Point and 90% confidence interval
    Values = row["17O/30Si"] * row["SiO2"] * calibration[:, 0] + calibration[:, 1]
    bestfit = Values[0]
    median = np.quantile(Values, 0.5)
    N_err = np.quantile(Values, 0.05) - np.quantile(Values, 0.5)
    P_err = np.quantile(Values, 0.95) - np.quantile(Values, 0.5)
    return [index, bestfit, median, N_err, P_err]


# %%
#
# Load NanoSIMS Summary Data and NanoSIMS points

Calibration_outputs.keys()


def calculate_H2O(DF, calibration_MC_dict, phase, date, confidence_interval=0.9):
    # Calculate Median Point and 90% confidence interval for H2O in phase for a given date
    # pyjanitor cleaning of dataset
    # DF = DF.remove_empty()
    DF = DF.replace("NaN", np.nan)
    DF = DF.replace(" ", np.nan)
    filenames = SIMS2020_DF.iloc[:, 0]
    slopes = calibration_MC_dict[(date, phase)]["slopes"].reshape(-1, 1)
    intercepts = calibration_MC_dict[(date, phase)]["intercepts"].reshape(-1, 1)

    H2O_MC = (
        DF["17O/30Si"].to_numpy().reshape(-1, 1) * DF["SiO2"].to_numpy().reshape(-1, 1)
    ) * slopes.T + intercepts.T
    quantile_low, quantile_high = [
        0.5 - confidence_interval / 2,
        0.5 + confidence_interval / 2,
    ]

    DF["bestfits" + "_" + phase] = H2O_MC[:, 0]
    medians = np.quantile(H2O_MC, 0.5, axis=1)
    DF["medians" + "_" + phase] = medians
    DF["N_errs_5%" + "_" + phase] = np.quantile(H2O_MC, quantile_low, axis=1) - medians
    DF["P_errs_95%" + "_" + phase] = (
        np.quantile(H2O_MC, quantile_high, axis=1) - medians
    )

    if "17O/30Si_Sigma" in DF.columns:
        DF["H2O SIMS_err ppm" + "_" + phase] = (
            DF["17O/30Si_Sigma"] * DF["SiO2"] * slopes[0, 0]
        )

    return DF  # [filenames, bestfits, medians, N_errs, P_errs]


# %%

SIMS2020_DF = pd.read_excel(
    excel_path,
    sheet_name="raw_data_mountA",
    # header=1,
    # index='Unique_ID',
    engine="openpyxl",
)

MountA_2020_samples = calculate_H2O(
    SIMS2020_DF, Calibration_outputs, "opx", "2020-02", confidence_interval=0.9
)
MountA_2020_samples.to_excel(
    "Caltech SIMS OPX and Olivine_With_Concentrations_2020-02_H2O_mount_A_Si_Norm.xlsx"
)


MountA_2020_samples = calculate_H2O(
    SIMS2020_DF, Calibration_outputs, "cpx", "2020-02", confidence_interval=0.9
)
MountA_2020_samples.to_excel(
    "Caltech SIMS CPX With_Concentrations_2020-02_H2O_mount_A_Si_Norm.xlsx"
)


# %%
SIMS2020_DF = pd.read_excel(
    excel_path,
    sheet_name="raw_data_mountB",
    # header=1,
    # index='Unique_ID',
    engine="openpyxl",
)

MountB_2020_samples = calculate_H2O(
    SIMS2020_DF, Calibration_outputs, "opx", "2020-02", confidence_interval=0.9
)
MountB_2020_samples.to_excel(
    "Caltech SIMS OPX and Olivine_With_Concentrations_2020-02_H2O_mount_B_Si_Norm.xlsx"
)


MountB_2020_samples = calculate_H2O(
    SIMS2020_DF, Calibration_outputs, "cpx", "2020-02", confidence_interval=0.9
)
MountB_2020_samples.to_excel(
    "Caltech SIMS CPX With_Concentrations_2020-02_H2O_mount_B_Si_Norm.xlsx"
)


# %%
