import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
# Cleaning the Price column during load as it contains commas and is interpreted as strings
df = pd.read_csv('/Users/hanimoudarres/Downloads/Foundations of ML/EDA/assignment/data/airbnb_hw.csv')
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

# 2. Dimensions, observations, variables, and head
print("Dimensions:", df.shape)
print("Variables:", df.columns.tolist())
print(df.head())

# 3. Cross tabulate Room Type and Property Type
print(pd.crosstab(df['Room Type'], df['Property Type']))

# 4. Price visualizations and description
print(df['Price'].describe())

plt.figure(figsize=(8, 5))
plt.hist(df['Price'].dropna(), bins=50, edgecolor='black')
plt.title("Histogram of Price")
plt.xlabel("Price ($)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.kdeplot(df['Price'].dropna(), fill=True)
plt.title("Kernel Density of Price")
plt.xlabel("Price ($)")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Price'].dropna())
plt.title("Box Plot of Price")
plt.xlabel("Price ($)")
plt.show()

# Log transform (replacing 0 with NaN to avoid log(0) errors)
df['price_log'] = np.log(df['Price'].replace(0, np.nan))
print(df['price_log'].describe())

plt.figure(figsize=(8, 5))
plt.hist(df['price_log'].dropna(), bins=50, edgecolor='black')
plt.title("Histogram of Log Price")
plt.xlabel("Log Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.kdeplot(df['price_log'].dropna(), fill=True)
plt.title("Kernel Density of Log Price")
plt.xlabel("Log Price")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['price_log'].dropna())
plt.title("Box Plot of Log Price")
plt.xlabel("Log Price")
plt.show()

# 5. Scatterplot and groupby beds
plt.figure(figsize=(8, 5))
plt.scatter(df['Beds'], df['price_log'], alpha=0.5)
plt.title("Scatterplot of Log Price vs Beds")
plt.xlabel("Beds")
plt.ylabel("Log Price")
plt.show()

print(df.groupby('Beds')['Price'].describe())

# 6. Scatterplot colored by room type and property type
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Beds', y='price_log', hue='Room Type', style='Property Type', alpha=0.7)
plt.title("Log Price vs Beds by Room & Property Type")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print(df.groupby(['Room Type', 'Property Type'])['Price'].describe())

# 7. Jointplot with hex
sns.jointplot(data=df, x='Beds', y='price_log', kind='hex', gridsize=20, cmap='Blues')
plt.show()