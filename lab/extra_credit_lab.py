import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1 & 2. Load Data and Select Variables
filepath = "/Users/hanimoudarres/Downloads/Foundations of ML/EDA/lab/Data/GSS.csv"
df_raw = pd.read_csv(filepath)

columns_of_interest = [
    'year', 'age', 'educ', 'polviews', 'realinc', 
    'happy', 'mntlhlth', 'trust', 'consci', 'gunlaw'
]
df = df_raw[columns_of_interest].copy()

# 3. Clean Data for EDA

# Safely force these to numeric. Any text (like '.i: Inapplicable') becomes NaN automatically.
df['mntlhlth'] = pd.to_numeric(df['mntlhlth'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')

# Remove negative error codes (like -100 or -97)
df['mntlhlth'] = df['mntlhlth'].apply(lambda x: np.nan if x < 0 else x)
df['realinc'] = df['realinc'].apply(lambda x: np.nan if x < 0 else x)

# Clean the remaining text/categorical columns, checking for both object AND modern string types
for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and (x.startswith('.') or x.startswith('-')) else x)

# Convert age 
df['age'] = df['age'].replace('89 or older', 89)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Map ordinal text variables to numerical scores for correlation
happy_map = {'Not too happy': 1, 'Pretty happy': 2, 'Very happy': 3}
df['happy_score'] = df['happy'].map(happy_map)

pol_map = {
    'Extremely liberal': 1, 'Liberal': 2, 'Slightly liberal': 3, 
    'Moderate, middle of the road': 4, 'Slightly conservative': 5, 
    'Conservative': 6, 'Extremely conservative': 7
}
df['polviews_score'] = df['polviews'].map(pol_map)

consci_map = {'HARDLY ANY': 1, 'ONLY SOME': 2, 'A GREAT DEAL': 3}
df['consci_score'] = df['consci'].map(consci_map)

# Extract years of education from the string format
def extract_educ(e):
    if pd.isna(e): return np.nan
    e = str(e)
    if '12th' in e: return 12
    import re
    m = re.search(r'\d+', e)
    if not m: return np.nan
    num = int(m.group())
    if 'college' in e: return 12 + num 
    return num

df['educ_years'] = df['educ'].apply(extract_educ)


# 4. Numeric Summaries and Visualizations

print("Numeric Summaries")
numeric_cols = ['age', 'realinc', 'mntlhlth', 'happy_score', 'polviews_score', 'educ_years', 'consci_score']
print(df[numeric_cols].describe())

# Visualization 1: Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix of GSS Variables')
plt.tight_layout()
plt.show()

# Visualization 2: Boxplot of Income vs Happiness
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='happy', y='realinc', order=['Not too happy', 'Pretty happy', 'Very happy'])
plt.title('Family Income Distribution by Happiness Level')
plt.xlabel('Happiness Level')
plt.ylabel('Real Family Income (Constant $)')
plt.show()

# Visualization 3: Trust in Science by Political Views
pol_consci = df.groupby('polviews')['consci_score'].mean().sort_values()
plt.figure(figsize=(10, 6))
pol_consci.plot(kind='barh', color='teal')
plt.title('Average Trust in Science by Political Ideology')
plt.xlabel('Trust in Science Score (1 = Hardly Any, 3 = A Great Deal)')
plt.ylabel('Political Views')
plt.tight_layout()
plt.show()

# Visualization 4: Mental Health by General Happiness
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='happy', y='mntlhlth', order=['Not too happy', 'Pretty happy', 'Very happy'])
plt.title('Poor Mental Health Days by Happiness Status')
plt.xlabel('Happiness Level')
plt.ylabel('Average Poor Mental Health Days (Past 30 Days)')
plt.show()