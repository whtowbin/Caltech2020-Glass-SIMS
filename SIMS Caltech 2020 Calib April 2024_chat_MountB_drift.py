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
    sheet_name="Mount B Glass Check Standards",
    # header=1,
    # index='Unique_ID',
    engine="openpyxl",
)

# pyjanitor cleaning of dataset
Calibrations = Calibrations.remove_empty()  # .dropna()

Calibrations.set_index("File", inplace=True)


# %%
# Define dictionaries for different elements
element_data = {
    "H2O": {
        "counts": "17O/30Si",
        "SiO2": "SiO2",
        "concentration": "H2O ppm",
        "concentration_sigma": "H2O_ppm_sigma",
        "counts_sigma": "17O/30Si_Sigma",
        "x_label": r"$\frac{^{16}OH}{^{30}Si}$ * SiO2 wt%",
    },
    "CO2": {
        "counts": "12C/30Si",
        "SiO2": "SiO2",
        "concentration": "CO2 ppm",
        "concentration_sigma": "CO2 ppm sigma",
        "counts_sigma": "12C/30Si_Sigma",
        "x_label": r"$\frac{^{12}C}{^{30}Si}$ * SiO2 wt%",
    },  # Add sigma if available
    "S": {
        "counts": "32S/30Si",
        "SiO2": "SiO2",
        "concentration": "S ppm",
        "counts_sigma": "32S/30Si_Sigma",
        "x_label": r"$\frac{^{32}S}{^{30}Si}$ * SiO2 wt%",
    },  # Add sigma if available
    "Cl": {
        "counts": "35Cl/30Si",
        "SiO2": "SiO2",
        "concentration": "Cl ppm",
        "counts_sigma": "35Cl/30Si_Sigma",
        "x_label": r"$\frac{^{35}Cl}{^{30}Si}$ * SiO2 wt%",
    },  # Add sigma if available
    "F": {
        "counts": "19F/30Si",
        "SiO2": "SiO2",
        "concentration": "F ppm",
        "counts_sigma": "19F/30Si_Sigma",
        "x_label": r"$\frac{^{19}F}{^{30}Si}$ * SiO2 wt%",
    },  # Add sigma if available
}

# {"C":{'counts':"12C/30Si", "SiO2":"SiO2", "concentration":"CO2 ppm"}}


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
# Plot Calinbration Lines for each element and save as a pdf
Phases = Calibrations.Phase.unique()
Calculated = {}
Calibration_outputs = {}
day = np.datetime64("2020-02")

pp = PdfPages("Caltech 2020 NS Calibrations_30Si_Norm_Drift Calib Mount B.pdf")

elements = ["H2O", "CO2", "S", "Cl", "F"]  # List of elements to calibrate

# Create subplots outside the loops, one for each element
fig, axs = plt.subplots(nrows=len(elements), ncols=1, figsize=(8.3, 30), dpi=300)
plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust spacing as needed

temp_dfs = {}

for idx, element in enumerate(elements):
    element_dict = element_data[element]  # get dictionary for element
    ax = axs[idx]  # Get the axis for the current element

    for phase in ["glass"]:  # Phases[2:3]:
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

            # Use the dictionary to get the correct column names
            X = Session[element_dict["counts"]] * Session[element_dict["SiO2"]]
            Y = Session[element_dict["concentration"]]

            # Check if sigma values exist, otherwise set to None

            if "concentration_sigma" in element_dict:
                sigma_y = Session[element_dict["concentration_sigma"]]
            else:
                sigma_y = Y * 0.01 + 10
            # sigma_y = (
            #     Session[element_dict["concentration_sigma"]]
            #     if "concentration_sigma" in element_dict
            #     else Y * 0.1 + 10
            # )
            sigma_x = (
                Session[element_dict["counts_sigma"]] * Session[element_dict["SiO2"]]
                if "counts_sigma" in element_dict
                else X * 0.1
            )

            # Create a temporary DataFrame and drop NA values
            temp_df = pd.DataFrame({"X": X, "Y": Y, "sigma_x": sigma_x, "sigma_y": sigma_y})

            temp_df = temp_df.dropna()

            X = temp_df["X"]
            Y = temp_df["Y"]

            sigma_y = temp_df["sigma_y"]

            sigma_x = (
                temp_df["sigma_x"] * 10
            )  # Multiply by 10 since sigma x is standard error of 100 sub_measurements

            temp_dfs[element] = temp_df

            # X, Y, sigma_y, sigma_x = (
            #     Session["17O/30Si"] * Session["SiO2"],
            #     Session["H2O ppm"],
            #     Session["H2O_ppm_sigma"],
            #     Session["17O/30Si_Sigma"],
            # )

            # plt.plot(X, Y, linestyle="none", marker="o")
            ax.errorbar(  # Use the current element's axis
                X,
                Y,
                yerr=sigma_y,
                xerr=sigma_x,
                linestyle="none",
                marker="o",
                ecolor="k",
                capsize=2,
                markeredgecolor="k",
                label=phase,  # Add a label for the phase
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
                ax=ax,  # Use the current element's axis
                LineMax=LineMax,
                LineInt=LineInterval,
            )
            # print(f"odr: {output[1]}")
            # Calibration_log[np.datetime_as_string(day, "M")] = output
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=-10)
            ax.set_ylabel(f"{element} (ppm)")  # Changed to dynamic element
            ax.set_xlabel(element_dict["x_label"])  # You might need to change this
            ax.set_title(f"{np.datetime_as_string(day, 'M')} - {element}")  # Added element to title

            # Cal_date_DF = Calibrations.loc[(Calibrations["Date"] == day)]
            # for index, row in Cal_date_DF.iterrows():
            #     Calculated[phase].append(
            #         calculate_values(index, row, np.array(output[3]))
            #     )

            slopes = np.array(output[3])[:, 0]
            intercepts = np.array(output[3])[:, 1]
            Calibration_outputs[
                (np.datetime_as_string(day, "M"), phase, element)
            ] = {  # Added element to Calibration outputs
                "slopes": slopes,
                "intercepts": intercepts,
            }

            ax.annotate(
                f"Y = {round(output[1][0])}X(±{round(np.std(slopes))}) + {round(output[1][1])}(±{round(np.std(intercepts))})",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
            )
            ax.annotate(
                rf"$R^{2}$ = {round(Linear_R2(X, Y, output[1][0], output[1][1]), 3)}",
                xy=(0.05, 0.90),
                xycoords="axes fraction",
            )

        except Exception as e:
            print("An Exception Occurred: ")
            traceback.print_exc()
            print(phase)
            print(day)
            print(e)
    ax.legend()  # Show legend for phases on each element plot

plt.show()
pp.savefig(fig)
pp.close()
# %%
