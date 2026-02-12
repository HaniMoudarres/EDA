import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
filename = '/Users/hanimoudarres/Downloads/Foundations of ML/EDA/assignment/data/ForeignGifts_edu.csv'
df = pd.read_csv(filename)

# --- Q2.2 Foreign Gift Amount ---
print("\n--- Q2.2 Foreign Gift Amount ---")
# Describe the variable
print(df['Foreign Gift Amount'].describe())

# Create a histogram
plt.figure(figsize=(10, 6))
# Using a log scale for the y-axis to better visualize the wide range of frequencies
plt.hist(df['Foreign Gift Amount'], bins=50, edgecolor='black')
plt.title('Histogram of Foreign Gift Amount')
plt.xlabel('Amount ($)')
plt.ylabel('Frequency (Log Scale)')
plt.yscale('log')
plt.show()

# --- Q2.3 Gift Type ---
print("\n--- Q2.3 Gift Type ---")
# Value counts and proportions
gift_counts = df['Gift Type'].value_counts()
gift_proportions = df['Gift Type'].value_counts(normalize=True)
print("Counts:\n", gift_counts)
print("Proportions:\n", gift_proportions)

# --- Q2.4 Kernel Density Plots ---
# Create log of Foreign Gift Amount
# We use numpy's log function. Note: This assumes amounts are positive. 
# The description showed a negative min, but for log plots we typically filter or handle <= 0.
# Here we filter for positive amounts for the log plot.
df_pos = df[df['Foreign Gift Amount'] > 0].copy()
df_pos['Log Foreign Gift Amount'] = np.log(df_pos['Foreign Gift Amount'])

# KDE of Log Amount
plt.figure(figsize=(10, 6))
sns.kdeplot(df_pos['Log Foreign Gift Amount'], fill=True)
plt.title('KDE of Log Foreign Gift Amount')
plt.xlabel('Log(Amount)')
plt.show()

# KDE Conditional on Gift Type
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_pos, x='Log Foreign Gift Amount', hue='Gift Type', common_norm=False)
plt.title('KDE of Log Foreign Gift Amount by Gift Type')
plt.show()

# --- Q2.5 Top 15 Countries ---
print("\n--- Q2.5 Top 15 Countries ---")
# Top 15 by number of gifts
top_countries_count = df['Country of Giftor'].value_counts().head(15)
print("Top 15 by Count:\n", top_countries_count)

# Top 15 by amount given
top_countries_amt = df.groupby('Country of Giftor')['Foreign Gift Amount'].sum().sort_values(ascending=False).head(15)
print("Top 15 by Amount:\n", top_countries_amt)

# --- Q2.6 Top 15 Institutions & Histogram ---
print("\n--- Q2.6 Top 15 Institutions ---")
# Total amount per institution
inst_totals = df.groupby('Institution Name')['Foreign Gift Amount'].sum()
top_inst = inst_totals.sort_values(ascending=False).head(15)
print("Top 15 Institutions by Amount:\n", top_inst)

# Histogram of total amounts received by all institutions
plt.figure(figsize=(10, 6))
plt.hist(inst_totals, bins=50, edgecolor='black')
plt.title('Histogram of Total Amounts Received by Institutions')
plt.xlabel('Total Amount Received ($)')
plt.ylabel('Number of Institutions')
plt.yscale('log') # Log scale for readability
plt.show()

# --- Q2.7 Top Giftors ---
print("\n--- Q2.7 Top Giftors ---")
top_giftors = df.groupby('Giftor Name')['Foreign Gift Amount'].sum().sort_values(ascending=False).head(15)
print("Top Giftors by Amount:\n", top_giftors)