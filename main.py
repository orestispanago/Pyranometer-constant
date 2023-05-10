import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


def read_files():
    csv_files =glob.glob("data/*.csv")
    df_list=[]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, parse_dates=True, index_col="Datetime_UTC")
        df = df.tz_localize("UTC")
        df_list.append(df)
    df_all = pd.concat(df_list)
    return df_all

def wm2_to_mv(wm2, constant):
    return wm2 * constant / 1000

def mv_to_wm2(mv, constant):
    return mv * 1000 / constant

def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created folder: {dir_path}")

def save_regresults(results, fname="out/linregress_stats.json"):
    linregress_dict = {
        "slope": results.params.GHI_mV,
        "slope_stderr": results.bse.values[0],
        "r2" : results.rsquared,
        "pvalue" : results.pvalues.values[0],
        "count":len(df),
        "outliers": len(outliers),
        "non-outliers" : len(good),
    }
    print(json.dumps(linregress_dict, indent=4))
    mkdir_if_not_exists(os.path.dirname(fname))
    with open(fname, "w") as f:
        json.dump(linregress_dict, f, indent=4)

def clean_IQR(df, test="GTI_mV", ref="GHI_mV"):
    resids =df[test]- df[ref]
    mbe=np.mean(resids)
    rmse = mean_squared_error(df[ref], df[test], squared=False)
    print("MBE:",mbe)
    print("RMSE:",rmse)
    q75 = np.percentile(resids, 75)
    q25 = np.percentile(resids, 25)
    iqr = q75 - q25  # InterQuantileRange
    is_good =(resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr))
    good = df[is_good]
    outliers = df[~is_good]
    print(f"Outliers: {len(outliers)} of {len(df)}")
    return good, outliers

df = read_files()
df = df.loc["2023-05-04 10:00:00":]

df["GHI_mV"] = wm2_to_mv(df["GHI_Avg"], 8.63)
df["GTI_mV"] = wm2_to_mv(df["GTI_Avg"], 5.12)


good, outliers = clean_IQR(df)
x, y = good["GHI_mV"], good["GTI_mV"]


model = sm.OLS(y,x)
results = model.fit()
save_regresults(results)

fig, ax = plt.subplots()
ax.scatter(x, y, s=14)
# ax.scatter(outliers["GHI_mV"], outliers["GTI_mV"], label="Outliers")
slope = results.params.GHI_mV
slope_stderr= results.bse.values[0]
r2 = results.rsquared
label = f"$y={slope:.5f}x \pm {slope_stderr:.1},\ R^2={r2:.2f}$"
ax.plot(x, slope * x, color="red", label=label)
ax.legend()
ax.set_xlabel("Reference $(mV)$")
ax.set_ylabel("Test $(mV)$")
plt.show()

print(f"Pyranometer constant: {results.params.GHI_mV *8.63:.2f}")

df["GTI_Avg_corr"] = mv_to_wm2(df["GTI_mV"],results.params.GHI_mV *8.63)
