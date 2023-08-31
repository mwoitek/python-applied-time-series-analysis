# pyright: analyzeUnannotatedFunctions=true

# %% [markdown]
# # ARIMA

# %%
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook

# %%
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [10, 7.5]

# %% [markdown]
# ## Import data

# %%
data = pd.read_csv("jj.csv")
data.head()

# %%
plt.plot(data["date"], data["data"])
plt.title("Quarterly EPS for J&J")
plt.xlabel("Date")
plt.ylabel("EPS ($)")
plt.xticks(rotation=90);  # fmt: skip

# %%
plot_acf(data["data"], auto_ylims=True);  # fmt: skip

# %%
plot_pacf(data["data"], auto_ylims=True);  # fmt: skip

# %%
data["data_tr_1"] = np.log(data["data"])
data["data_tr_1"] = data["data_tr_1"].diff()
data.head(10)

# %%
plt.plot(data["data_tr_1"])
plt.title("Log Difference of Quarterly EPS for J&J");  # fmt: skip

# %%
ad_fuller_result = adfuller(data["data_tr_1"][1:])
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

# %%
plot_acf(data["data_tr_1"][1:], auto_ylims=True);  # fmt: skip

# %%
plot_pacf(data["data_tr_1"][1:], auto_ylims=True);  # fmt: skip


# %%
def optimize_ARIMA(endog, order_list):
    """
    Returns a DataFrame with parameters and corresponding AIC

    endog - The observed variable
    order_list - List with (p, d, q) tuples
    """
    results = []
    for order in tqdm_notebook(order_list):
        try:
            model = SARIMAX(
                endog,
                order=order,
                simple_differencing=False,
            ).fit(disp=False)
        except:
            continue
        results.append([order, model.aic])

    results_df = pd.DataFrame(results)
    results_df.columns = ["(p, d, q)", "AIC"]
    results_df = results_df.sort_values(by="AIC").reset_index(drop=True)
    return results_df


# %%
ps = range(8)
d = 1
qs = range(8)

order_list = list(map(lambda t: (t[0], d, t[1]), product(ps, qs)))
order_list[:10]

# %%
results_df = optimize_ARIMA(data["data"], order_list)
results_df

# %%
best_model = SARIMAX(
    data["data"], order=(6, 1, 3), simple_differencing=False
)
res = best_model.fit(disp=False)
res.summary()

# %%
res.plot_diagnostics();  # fmt: skip

# %%
n_forecast = 8
predict = res.get_prediction(end=best_model.nobs + n_forecast)
idx = np.arange(len(predict.predicted_mean))

fig, ax = plt.subplots()
ax.plot(data["data"], "blue")
ax.plot(idx[-n_forecast:], predict.predicted_mean[-n_forecast:], "k--")
ax.set_title("Forecast of Quarterly EPS for J&J");  # fmt: skip

# %%
data["model"] = predict.predicted_mean
data.head(15)

# %%
mse = mean_squared_error(data["data"], data["model"])
print(f"MSE: {mse}")
