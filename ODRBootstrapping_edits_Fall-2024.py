# %%
"""
https://micropore.wordpress.com/2017/02/07/python-fit-with-error-on-both-axis/
Code is modified after the above blog post
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from pylab import *
from scipy import odr
from scipy.optimize import curve_fit

# %%


def ODR_Linear(x, y, x_err, y_err, intercept=False, InitialGuess=[100, 1]):
    def yint_func(p, x):
        # Line fit with slope and y intercept.
        a, b = p
        return a * x + b

    def slope_func(p, x):
        # Line with slope fixed through zero
        a = p
        return a * x

    linear_model = odr.Model(yint_func)

    # Model object
    if intercept is False:
        linear_model = odr.Model(slope_func)

        # if len(InitialGuess) > 1:
        #   raise Exception(
        #        "If Intercept is False: Initial Guess should only have slope"
        #   )

    data = odr.RealData(x, y, sx=x_err, sy=y_err)
    # Set up ODR with the model and data.
    myodr = odr.ODR(data, linear_model, beta0=InitialGuess)
    # THis might need to be set back to zero after this test

    myodr.set_job(fit_type=0)
    # Run the regression.
    out = myodr.run()

    # print fit parameters and 1-sigma estimates
    Popt = out.beta
    Perr = out.sd_beta
    return (Popt, Perr)


#
def ODR_Linear_Test(x, y, x_err, y_err, intercept=False, InitialGuess=[100, 1]):
    def yint_func(p, x):
        # Line fit with slope and y intercept.
        a, b = p
        return a * x + b

    def slope_func(p, x):
        # Line with slope fixed through zero
        a = p
        return a * x

    linear_model = odr.Model(yint_func)

    # Model object
    if intercept is False:
        linear_model = odr.Model(slope_func)

        # if len(InitialGuess) > 1:
        #   raise Exception(
        #        "If Intercept is False: Initial Guess should only have slope"
        #   )

    data = odr.RealData(x, y, sx=x_err, sy=y_err)
    # Set up ODR with the model and data.
    myodr = odr.ODR(data, linear_model, beta0=InitialGuess)
    # THis might need to be set back to zero after this test

    myodr.set_job(fit_type=0)
    # Run the regression.
    out = myodr.run()

    # print fit parameters and 1-sigma estimates
    Popt = out.beta
    Perr = out.sd_beta
    return (Popt, Perr, out)


# %%


def Bootstrap_fit(
    x, y, x_err, y_err, resample_draws, InterceptFit=True, InitialGuess=[100, 1]
):
    def resample(l):
        return np.random.randint(0, l, l)

    if InterceptFit is False:
        InitialGuess = [InitialGuess[0]]
        # Maybe make this just the first value of the input so it has more flexibility.

    # data = [x, x_err, y, y_err]
    # df = pd.DataFrame(data, index=["x", "x_err", "y", "y_err"]).T
    data = np.array([x, x_err, y, y_err]).T
    df = pd.DataFrame(data, columns=["x", "x_err", "y", "y_err"])

    df.dropna(inplace=True)
    length = len(df)

    opt, err = ODR_Linear(
        x=df["x"],
        y=df["y"],
        x_err=df["x_err"],
        y_err=df["y_err"],
        InitialGuess=InitialGuess,
        intercept=InterceptFit,
    )
    Fit_Param = [opt]
    subs = []
    # Sub added here to test what the output is

    for idx in range(resample_draws):
        sub = df.take(resample(length))
        opt, err = ODR_Linear(
            x=sub["x"],
            y=sub["y"],
            x_err=sub["x_err"],
            y_err=sub["y_err"],
            InitialGuess=InitialGuess,
            intercept=InterceptFit,
        )
        Fit_Param.append(opt)
        subs.append(sub)
    return Fit_Param, subs


# %%
def yint_func(p, x):
    # Line fit with slope and y intercept.
    a, b = p
    return a * x + b


def slope_func(p, x):
    # Line fit with through 0
    a = p
    return a * x


def Eval_Conf(Fit_Param, Confidence_Bound=0.95, LineMax=200, LineInt=1, **kwargs):
    """Funtion evaluates the bootstrapped linear regressions and determines 95% confidence intervals.

    Arguments:
        Fit_Param {} -- Iterable of Slopes and or Intercepts

    Keyword Arguments:
        Max {int} -- [The number of points in the line] (default: {200})

    Returns:
        [DataFrame] -- Columns: best_fit Line, neg_error_bound, pos_error_bound(95%)
    """
    if len(Fit_Param[0]) > 2:
        raise exception(
            "Fit_Param has too many inputs per row. Line inputs must be 1 or 2 parameters"
        )
    FitFunc = yint_func
    if len(Fit_Param[0]) is 1:
        FitFunc = slope_func

    evaluted = []
    x = np.arange(0, LineMax, LineInt)
    for row in Fit_Param:
        evaluted.append(FitFunc(row, x))

    BootStp_Samples = pd.DataFrame(evaluted)
    confidence_ints = []
    for (ColName, col) in BootStp_Samples.items():
        histrange = (np.nanmin(col), np.nanmax(col))
        hist = np.histogram(col, bins=200, range=histrange)
        conf_int = stats.rv_histogram(hist).interval(Confidence_Bound)
        confidence_ints.append(conf_int)

    Results = pd.DataFrame(
        confidence_ints, columns=("neg_error_bound", "pos_error_bound")
    )

    # index should be added as X axis
    Results.index = x
    Results["best_fit"] = evaluted[0]
    Results["percent_error_neg"] = (
        Results["best_fit"] - Results["neg_error_bound"]
    ) / np.abs(Results["best_fit"])
    Results["percent_error_pos"] = (
        Results["pos_error_bound"] - Results["best_fit"]
    ) / np.abs(Results["best_fit"])
    return Results


# %%

# this needs to be set up to input a dictionary
def plot_regression(
    confidence_df,
    datapoints=None,
    LineMax=200,
    LineInt=1,
    ax=None,
    ecolor="r",
    line_color="b",
    sigma=2,
    e_alpha=0.5,
    **kwargs,
):

    BestFitLine = confidence_df["best_fit"]
    NegBound = confidence_df["neg_error_bound"]
    PosBound = confidence_df["pos_error_bound"]

    x = NegBound.index
    if ax is None:
        ax = plt.gca()

    ax.fill_between(x, NegBound, PosBound, color=ecolor, alpha=e_alpha)
    ax.plot(x, BestFitLine, color=line_color, **kwargs)

    # ax.fill_between(x, NegBound, PosBound, color=ecolor, alpha=e_alpha)

    if datapoints is not None:
        ax.errorbar(
            x=datapoints["x"],
            y=datapoints["y"],
            yerr=datapoints["yerr"],
            xerr=datapoints["xerr"],
            marker=".",
            fmt="g",
            linestyle="none",
            capsize=5,
            markeredgewidth=1,
            markersize=10,
            label=None,
            **kwargs,
        )

    return ax


# %%


def ODR_Bootstrap(
    x,
    y,
    x_err,
    y_err,
    resample_draws=5000,
    LineMax=200,
    LineInterval=1,
    InterceptFit=True,
    InitialGuess=[100, 1],
    Confidence_Bound=0.95,
    plot=False,
    ax=None,
    **kwargs,
):

    """[summary]

    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
        x_err {[type]} -- [description]
        y_err {[type]} -- [description]

    Keyword Arguments:
        resample_draws {int} -- [description] (default: {2000})
        LineMax {int} -- [description] (default: {200})
        LineInterval {int} -- [description] (default: {1})
        InterceptFit {bool} -- [description] (default: {True})
        InitialGuess {list} -- [description] (default: {[100, 1]})
        plot {bool} -- [description] (default: {False})
        ax {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """

    param, subs = Bootstrap_fit(
        x, y, x_err, y_err, resample_draws, InterceptFit, InitialGuess
    )
    confidence_data = Eval_Conf(
        Fit_Param=param,
        Confidence_Bound=Confidence_Bound,
        LineMax=LineMax,
        LineInt=LineInterval,
    )

    points = pd.DataFrame({"x": x, "y": y, "xerr": x_err, "yerr": y_err})

    points.dropna(inplace=True)

    return confidence_data, param[0], points, param, subs


# %%
#
#
def gauss_agv_err(concentrations, errors, cut_off=0.000001):
    # Currently uses masked arrays which is too slow.

    def gaussian(x, sigma, avg):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * (
            np.exp(-0.5 * ((x - avg) / sigma) ** 2)
        )

    def CI_bound(xi, data, bound_fraction):
        for n, val in enumerate(np.cumsum(data)):
            if val > bound_fraction:
                return round(xi[n], 2)

    def find_range(avgs, sigmas):
        max = np.max(avgs) + 3 * np.max(sigmas)
        min = np.min(avgs) - 3 * np.max(sigmas)
        return (min, max)

    min, max = find_range(concentrations, errors)
    xi = np.arange(min, max, 0.01)
    x = np.tile(xi, (len(concentrations), 1))
    unnormed_data = np.sum(gaussian(x.T, errors, concentrations), axis=1)
    data = unnormed_data / np.trapz(unnormed_data)

    # data = np.ma.masked_where(data < cut_off, data)
    # xi = np.ma.masked_where(data < cut_off, xi)

    # average = np.dot(np.ma.compressed(xi), np.ma.compressed(data)) / np.sum(
    #     np.ma.compressed(data)
    # )
    average = np.dot(xi, data) / np.sum(data)

    most_frequent = xi[np.argmax(data)]
    # mean = np.mean(concentrations)
    best_fit = concentrations[0]
    center_of_mass = CI_bound(xi, data, 0.50)
    one_sigma_bounds = CI_bound(xi, data, 0.16), CI_bound(xi, data, 0.84)
    two_sigma_bounds = CI_bound(xi, data, 0.05), CI_bound(xi, data, 0.95)
    CI_one_sigma = round(center_of_mass - one_sigma_bounds[0], 2), round(
        one_sigma_bounds[1] - center_of_mass, 2
    )
    CI_two_sigma = round(center_of_mass - two_sigma_bounds[0], 2), round(
        two_sigma_bounds[1] - center_of_mass, 2
    )

    return {"x": xi, "y": data}, {
        "simple_best_fit": best_fit,
        "mean": average,
        "mode": most_frequent,
        "mid_point": center_of_mass,
        "one_sigma_bounds": one_sigma_bounds,
        "two_sigma_bounds": two_sigma_bounds,
        "CI_one_sigma": CI_one_sigma,
        "CI_two_sigma": CI_two_sigma,
        "n": len(concentrations),
    }


def plot_datapoints(data, bounds, ax=None, sample_name=None):
    ax = ax or plt.gca()
    # fig = ax.figure(figsize = (12,6),)
    x = data["x"]
    y = data["y"]  # /np.sum(data['y'])
    ax.plot(
        x,
        y,
        linewidth=3,
    )
    ax.set_xlabel("Concentration ppm")
    ax.set_ylabel("Probablility")

    # ax.title(sample_name + '____ Hydrgen SIMS Measurments ')
    # ax.annotate('')
    ax.axvline(
        x=bounds["mean"],
        ymin=0,
        color="b",
        linestyle="dashed",
        linewidth=3,
        label="Mean",
    )
    ax.axvline(
        x=bounds["mid_point"],
        ymin=0,
        color="g",
        linestyle="dashed",
        linewidth=3,
        label="Mid_point & 65% CI",
    )
    # ax.axvline(x=bounds['one_sigma_bounds'][0], ymin = 0, color = 'r', linestyle = 'dashed', linewidth = 2, label = '68% CI')
    # ax.axvline(x=bounds['one_sigma_bounds'][1], ymin = 0, color = 'r', linestyle = 'dashed', linewidth = 2)

    CI_one_sigma = bounds["CI_one_sigma"]
    CI_two_sigma = bounds["CI_two_sigma"]

    ax.annotate(
        f"""
    Simple Best Fit: { float(bounds['simple_best_fit']):.2f}
    Mean: { float(bounds['mean']):.2f} 
    Mode: { float(bounds['mode']):.2f}
    Mid-point:  { float(bounds['mid_point']):.2f}
    Confidence Intervals
    68%: - {CI_one_sigma[0]:.2f} / +{CI_one_sigma[1]:.2f}
    95%: - {CI_two_sigma[0]:.2f} / +{CI_two_sigma[1]:.2f}
    n: {bounds['n']}
    """,
        xy=(0.02, 0.68),
        xycoords="axes fraction",
        bbox=dict(boxstyle="square", fc="w", alpha=0.85),
    )
    eb = ax.errorbar(
        x=bounds["mid_point"],
        y=np.max(y) / 2,
        xerr=np.array([[CI_one_sigma[0]], [CI_one_sigma[1]]]),
        capsize=10,
        elinewidth=3,
        capthick=3,
        ecolor="g",
        linestyle="dashed",
    )
    eb[-1][0].set_linestyle("dashed")

    ax.set_ylim(
        bottom=0,
    )
    ax.legend(loc="upper right", framealpha=0.85)
    return ax


# %%


def plot_Calibration_Estimates(fit_params, fit_error, Title="Calibration Line Fits"):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    Slope_Fit_Params = gauss_agv_err(
        np.array(fit_params)[:, 0], np.array(fit_error)[:, 0]
    )

    plot_datapoints(Slope_Fit_Params[0], Slope_Fit_Params[1], ax=ax1)

    ax1.set_xlabel("Calibration Slope", fontsize=(20))
    ax1.set_ylabel("Probablity", fontsize=(20))

    Intercept_Fit_Params = gauss_agv_err(
        np.array(fit_params)[:, 1], np.array(fit_error)[:, 1]
    )

    plot_datapoints(Intercept_Fit_Params[0], Intercept_Fit_Params[1], ax=ax2)
    ax2.set_xlabel("Calibration Y-Intercept ppm", fontsize=(20))
    ax2.set_ylabel("Probablity", fontsize=(20))
    plt.suptitle(Title, fontsize=20)
    w_pad = 1.0
    fig.tight_layout()
    return fig
