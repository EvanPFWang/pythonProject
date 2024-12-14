# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.express as px
from datetime import datetime

# Load data
region = pd.read_csv("http://lenkiefer.com/img/charts_feb_20_2017/region.txt", delimiter='\t')
cbsa = pd.read_csv("http://lenkiefer.com/img/charts_feb_20_2017/cbsa.city.txt", delimiter='\t')
dt = pd.read_csv("http://www.freddiemac.com/fmac-resources/research/docs/fmhpi_master_file.csv")

# Define the function for color palettes
my_colors = {
    "green": (103/256, 180/256, 75/256),
    "green2": (147/256, 198/256, 44/256),
    "lightblue": (9/256, 177/256, 240/256),
    "blue": "#00aedb",
    "red": "#d11141",
    "orange": "#f37735",
    "yellow": "#ffc425",
    "gold": "#FFD700",
    "light grey": "#cccccc",
    "dark grey": "#8c8c8c"
}

# Add date columns and transformations
dt['date'] = pd.to_datetime(dt[['Year', 'Month']].assign(DAY=1))
dt['hpa12'] = dt.groupby('GEO_Name')['Index_SA'].transform(lambda x: x / x.shift(12) - 1)
dt['hpa3'] = dt.groupby('GEO_Name')['Index_SA'].transform(lambda x: (x / x.shift(3)) ** 4 - 1)
dt['decade'] = dt['date'].dt.year // 10 * 10

# Load CPI data
df_cpi = pd.read_csv("https://fred.stlouisfed.org/data/CUSR0000SA0L2.txt", delimiter=',', names=["date", "cpi"])
df_cpi['date'] = pd.to_datetime(df_cpi['date'])
df_cpi['cpi12'] = df_cpi['cpi'] / df_cpi['cpi'].shift(12) - 1

# Merge CPI data with the main dataset
dt = dt.merge(df_cpi, on="date", how="left")
dt['cpi1'] = dt.groupby('GEO_Name')['cpi'].transform(lambda x: x / x.iloc[0])
dt['rhpi'] = dt['Index_SA'] / dt.groupby('GEO_Name')['Index_SA'].transform('first') / (dt['cpi1'] / dt.groupby('GEO_Name')['cpi1'].transform('first'))

# Filter data for plotting
dt_msa = dt[dt['GEO_Type'] == 'CBSA']
dt_msa['max_hpi'] = dt_msa.groupby('GEO_Name')['Index_SA'].transform('max')
dt_msa['last_hpi'] = dt_msa.groupby('GEO_Name')['Index_SA'].transform('last')
dt_state = dt[dt['GEO_Type'] == 'State']
dt_state['max_hpi'] = dt_state.groupby('GEO_Name')['Index_SA'].transform('max')
dt_state['last_hpi'] = dt_state.groupby('GEO_Name')['Index_SA'].transform('last')

# Plotting: Time Series for USA
plt.figure(figsize=(10, 6))
usa_data = dt[dt['GEO_Name'] == "USA"]
plt.plot(usa_data['date'], usa_data['rhpi'], label="Real House Price Index", linewidth=2)
plt.scatter(usa_data['date'].iloc[-1], usa_data['rhpi'].iloc[-1], color='red', zorder=5)
plt.axhline(usa_data['rhpi'].iloc[-1], color='red', linestyle='--', alpha=0.6)
plt.title("Real House Price for USA")
plt.xlabel("Date (monthly)")
plt.ylabel("Real House Price (Jan 2000 = 1)")
plt.grid()
plt.show()

# Plotting: Comparison of States
states_data = dt_state[dt_state['GEO_Name'].isin(['VA', 'CA', 'TX', 'OH'])]
plt.figure(figsize=(10, 6))
for state in states_data['GEO_Name'].unique():
    state_data = states_data[states_data['GEO_Name'] == state]
    plt.plot(state_data['date'], state_data['rhpi'], label=state, linewidth=2)
plt.legend(title="State")
plt.title("Real House Price in VA, CA, TX, and OH")
plt.xlabel("Date (monthly)")
plt.ylabel("Real House Price (Jan 2000 = 1)")
plt.grid()
plt.show()

# Scatterplot: Real House Price by State
scatter_data = dt_state[dt_state['Month'] == 3]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=scatter_data, x='rhpi', y='max_hpi', hue='region', style='region', s=100)
plt.axvline(1, color='red', linestyle='--')
plt.title("Real House Price by State")
plt.xlabel("Real House Price (1975-2006 max = 1)")
plt.ylabel("Real House Price (Jan 2000 = 1)")
plt.grid()
plt.show()

# Animated scatterplot by state
fig = px.scatter(
    scatter_data,
    x="rhpi",
    y="max_hpi",
    color="region",
    animation_frame="Year",
    hover_name="GEO_Name",
    title="Real House Price by State (Animated)",
    labels={"rhpi": "Real House Price (1975-2006 max = 1)", "max_hpi": "Real House Price (Jan 2000 = 1)"}
)
fig.show()

# Beeswarm Plot (using stripplot for simplicity)
plt.figure(figsize=(12, 8))
sns.stripplot(data=dt_msa[dt_msa['date'].dt.month == 3], x='region', y=np.log(dt_msa['rhpi']), jitter=True, palette='coolwarm')
plt.title("Real House Price by Metro Area (March 2022)")
plt.xlabel("Region")
plt.ylabel("Real House Price (log scale)")
plt.grid()
plt.show()
