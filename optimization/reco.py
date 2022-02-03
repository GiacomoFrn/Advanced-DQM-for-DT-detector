import pandas as pd
from pandas.core.indexes.numeric import IntegerIndex
import numpy as np
from scipy import optimize
from numpy import sqrt


# *****************************************************************************
# LINEAR_REG
# *****************************************************************************


def chisq(X, Y, SY, a, b):
    return sum(((y - a - x * b) / sy) ** 2 for x, y, sy in zip(X, Y, SY))


def fitfunc(x, a, b):
    return a + b * x


def linear_reg(X, Y):

    sigma_X = [0.4] * len(X)  # 400 um std
    mGuess = 100 if ((X[0] - X[1]) == 0) else (Y[0] - Y[1]) / (X[0] - X[1])
    qGuess = Y[0] - mGuess * X[0]
    p_init = [qGuess, mGuess]  # valori iniziali
    p_best, pcov = optimize.curve_fit(fitfunc, Y, X, sigma=sigma_X, p0=p_init)

    chi2 = chisq(Y, X, sigma_X, p_best[0], p_best[1])
    dof = len(X) - 2  # - 1
    chisq_comp = abs(chi2 - dof) / sqrt(2 * dof)

    m = p_best[1]
    q = p_best[0]
    return {"m": m, "q": q, "chisq_comp": chisq_comp}


# *****************************************************************************
# COMBINATE LOCAL
# *****************************************************************************

from itertools import combinations


def compute(df):  # DA OTTIMIZZARE E SNELLIRE
    
    comb = []
    if len(df.LAYER.unique()) == 3:
        comb.append(df)
        tot_Hits = 3
    else:
        for index in list(combinations(df.index, 4)):
            tmp_df = df.loc[index, :]
            if len(tmp_df.LAYER.unique()) == 4:
                comb.append(tmp_df)  # comb[] contains combinations of data
        tot_Hits = 4

    flag = True

    for data in comb:
        X = np.array(pd.concat([data["X_RIGHT_GLOB"], data["X_LEFT_GLOB"]]))
        Y = np.array(pd.concat([data["WIRE_Z_GLOB"], data["WIRE_Z_GLOB"]]))
        for indexes_comb in list(combinations(range(len(X)), tot_Hits)):
            indexes_comb = list(indexes_comb)
            if len(np.unique(X[indexes_comb])) == tot_Hits:
                regr_dict = linear_reg(X[indexes_comb], Y[indexes_comb])
                if flag:
                    min_lambda = abs(regr_dict["chisq_comp"])
                    xdata = X[indexes_comb]
                    res_dict = regr_dict
                    flag = False
                    best_comb = indexes_comb
                    best_data = data
                elif abs(regr_dict["chisq_comp"]) < min_lambda:
                    min_lambda = abs(regr_dict["chisq_comp"])
                    xdata = X[indexes_comb]
                    res_dict = regr_dict
                    best_comb = indexes_comb
                    best_data = data

    big_df = pd.concat([best_data, best_data], axis=0, ignore_index=True)
    reco_df = big_df.loc[best_comb, :]
    reco_df["m"] = np.full(len(reco_df), res_dict["m"])
    reco_df["q"] = np.full(len(reco_df), res_dict["q"])
    reco_df["X"] = xdata
    if xdata is None: return

    return reco_df


# *****************************************************************************
# COMPUTE EVENT
# *****************************************************************************

def computeEvent(df_E):

    chamber = [df_E[df_E["SL"] == i] for i in range(4)]
    event_reco_df = pd.DataFrame()

    for df in chamber:
        if len(pd.unique(df.LAYER)) < 3:
            continue

        chamber_reco_df = compute(df)
        event_reco_df = pd.concat(
            [event_reco_df, chamber_reco_df], axis=0, ignore_index=True
        )

    if (event_reco_df is None):
        return

    return event_reco_df


# *****************************************************************************
# GET RESULTS
# *****************************************************************************


def getRecoResults(events):
    resultsDf = []

    for df_E in events:
        if len(df_E) > 32:
            continue
        event_reco_df = computeEvent(df_E)
        if event_reco_df is None:
            continue
        resultsDf.append(event_reco_df)

    return resultsDf
