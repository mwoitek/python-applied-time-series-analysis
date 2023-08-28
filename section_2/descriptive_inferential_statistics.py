# %% [markdown]
# # Descriptive and Inferential Statistics

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# %%
plt.rcParams["figure.figsize"] = [10, 7.5]

# %% [markdown]
# ## Descriptive Statistics
# ### Dataset

# %%
data = pd.read_csv("shampoo.csv")
data.head()

# %% [markdown]
# ### Summary

# %%
data.describe()

# %%
data.mean(numeric_only=True)

# %%
data.std(numeric_only=True)

# %% [markdown]
# ### Visualizations
# #### Histograms

# %%
data.hist()

# %%
plt.hist(data["Sales"])
plt.title("Histogram of Shampoo Sales")
plt.xlabel("Shampoo Sales ($M)")
plt.ylabel("Frequency")

# %%
plt.hist(data["Sales"], bins=20, color="#fcba03")
plt.title("Histogram of Shampoo Sales")
plt.xlabel("Shampoo Sales ($M)")
plt.ylabel("Frequency")

# %%
# DEPRECATED FUNCTION
sns.distplot(
    data["Sales"],
    bins=20,
    hist=True,
    kde=True,
    color="#fcba03",
)
plt.title("Histogram of Shampoo Sales")
plt.xlabel("Shampoo Sales ($M)")
plt.ylabel("Density")

# %% [markdown]
# #### Scatter plots

# %%
sns.scatterplot(x=data["Month"], y=data["Sales"])
plt.title("Historical Sales of Shampoo")
plt.xlabel("Month")
plt.ylabel("Sales of Shampoo ($M)")

# %% [markdown]
# ## Inferential Statistics

# %%
co2_data = pd.read_csv("co2_dataset.csv")
co2_data.head()

# %%
X = co2_data["year"].values[1950:]
y = co2_data["data_mean_global"].values[1950:]

sns.scatterplot(x=X, y=y)
plt.title("Historical Global CO2 Concentration in the Atmosphere")
plt.xlabel("Year")
plt.ylabel("CO2 Concentration (ppm)")

# %%
X = X.reshape((-1, 1))
y = y.reshape((-1, 1))

reg = LinearRegression()
reg = reg.fit(X, y)
print(
    f"The slope is {reg.coef_[0][0]} and the intercept is {reg.intercept_[0]}"
)

# %%
X = co2_data["year"].values[1950:]
y = co2_data["data_mean_global"].values[1950:]

predictions = reg.predict(X.reshape((-1, 1)))

plt.scatter(X, y, c="black")
plt.plot(X, predictions, c="blue", linewidth=2)
plt.title("Historical CO2 Concentration")
plt.xlabel("Year")
plt.ylabel("CO2 Concentration (ppm)")
plt.show()

# %%
X = sm.add_constant(co2_data["year"].values[1950:])
model = sm.OLS(co2_data["data_mean_global"].values[1950:], X).fit()
model.summary()

# %%
residuals = model.resid
qq_plot = sm.qqplot(residuals, line="q")
plt.show()

# %%
plt.hist(residuals)
