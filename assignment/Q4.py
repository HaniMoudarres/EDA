import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Q4.1 Load the drilling_rigs.csv data
filepath = '/Users/hanimoudarres/Downloads/Foundations of ML/EDA/assignment/data/drilling_rigs.csv'
df = pd.read_csv(filepath)

# Examine the data
print("Dimensions:", df.shape)
print("Columns:", df.columns.tolist())
print("\nData Types Before Cleaning:\n", df.dtypes)
print(df.head())

# Clean the data by replacing 'Not Available' with NaN and coercing to numeric
df = df.replace('Not Available', np.nan)
for col in df.columns:
    if col != 'Month':
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nData Types After Cleaning:\n", df.dtypes)

# Q4.2 Convert Month variable to ordered datetime
df['time'] = pd.to_datetime(df['Month'], format='mixed')
df = df.sort_values('time')

# Q4.3 Line plot of Active Well Service Rig Count
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['Active Well Service Rig Count (Number of Rigs)'])
plt.title('Active Well Service Rig Count Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Active Rigs')
plt.show()

# Q4.4 Compute first difference and plot
df['rig_count_diff'] = df['Active Well Service Rig Count (Number of Rigs)'].diff()

plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['rig_count_diff'])
plt.title('First Difference of Active Well Service Rig Count')
plt.xlabel('Time')
plt.ylabel('Month-over-Month Change in Rigs')
plt.show()

# Q4.5 Melt first two columns (Onshore and Offshore) and plot
# Identifying the specific names for the onshore and offshore columns
onshore_col = 'Crude Oil and Natural Gas Rotary Rigs in Operation, Onshore (Number of Rigs)'
offshore_col = 'Crude Oil and Natural Gas Rotary Rigs in Operation, Offshore (Number of Rigs)'

df_melted = df.melt(
    id_vars=['time'], 
    value_vars=[onshore_col, offshore_col],
    var_name='Rig Location', 
    value_name='Count'
)

# Renaming the values for a much cleaner legend
df_melted['Rig Location'] = df_melted['Rig Location'].map({
    onshore_col: 'Onshore Rigs',
    offshore_col: 'Offshore Rigs'
})

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x='time', y='Count', hue='Rig Location')
plt.title('Onshore vs. Offshore Rigs Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Rigs')
plt.show()