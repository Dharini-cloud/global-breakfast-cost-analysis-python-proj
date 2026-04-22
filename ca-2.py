import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("breakfast basket.csv")

# -------------------------------
# 2. DATA CLEANING
# -------------------------------

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop fully empty rows
df = df.dropna(how='all')

# Separate columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# Handle missing values
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Remove constant columns (important)
nunique = df[num_cols].nunique()
constant_cols = nunique[nunique <= 1].index
df = df.drop(columns=constant_cols)

# Update numeric columns
num_cols = df.select_dtypes(include=np.number).columns

# -------------------------------
# 3. EDA
# -------------------------------
print("\nData Info:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nSkewness:\n")
print(df.skew(numeric_only=True))

# -------------------------------
# 4. VISUALIZATION (IMPROVED)
# -------------------------------
sns.set(style="whitegrid")

# 1. Histogram
plt.figure(figsize=(6,4))
sns.histplot(df['breakfast_basket_usd'], kde=True)
plt.title("Distribution of Breakfast Basket Cost")
plt.show()

# 2. Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=df['breakfast_basket_usd'])
plt.title("Outlier Detection (Basket Cost)")
plt.show()

# 3. Top Items by Cost
top_items = df.groupby('item')['breakfast_basket_usd'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
top_items.plot(kind='bar')
plt.title("Top 10 Items by Average Basket Cost")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x='price_usd', y='breakfast_basket_usd', data=df)
plt.title("Price vs Basket Cost")
plt.show()

# 5. Correlation Heatmap (clean)
important_cols = [
    'price_usd',
    'exchange_rate',
    'yoy_inflation_estimate_pct',
    'population_estimate',
    'breakfast_basket_usd'
]

plt.figure(figsize=(7,5))
sns.heatmap(df[important_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 5. CORRELATION (BEFORE)
# -------------------------------
num_df = df[num_cols]
print("\nCorrelation Matrix (Before):\n", num_df.corr())

# -------------------------------
# 6. IQR OUTLIER HANDLING (FIXED)
# -------------------------------
df_iqr = num_df.copy()

mask = pd.Series(True, index=df_iqr.index)

for col in num_cols:
    Q1 = df_iqr[col].quantile(0.25)
    Q3 = df_iqr[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask &= (df_iqr[col] >= lower) & (df_iqr[col] <= upper)

df_iqr = df_iqr[mask]

print("\nCorrelation After IQR:\n", df_iqr.corr())

# -------------------------------
# 7. Z-SCORE OUTLIER HANDLING
# -------------------------------g
z_scores = np.abs(zscore(num_df, nan_policy='omit'))

df_z = num_df[(z_scores < 3).all(axis=1)]

print("\nCorrelation After Z-score:\n", df_z.corr())

# -------------------------------
# LOAD CLEAN DATA (use your cleaned df)
# -------------------------------
df = pd.read_csv("breakfast basket.csv")

# Basic cleaning (minimal required)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.dropna().drop_duplicates().reset_index(drop=True)

# -------------------------------
# OBJECTIVE 4: HYPOTHESIS TESTING
# -------------------------------
print("\n===== HYPOTHESIS TESTING =====")

# Normality test (Shapiro)
sample_data = df['price_usd'].sample(500, random_state=42)
stat, p = stats.shapiro(sample_data)

print("Shapiro p-value:", p)

if p > 0.05:
    print("Data is normally distributed")
else:
    print("Data is NOT normally distributed")

# T-test (split data based on median)
median_val = df['price_usd'].median()

group1 = df[df['price_usd'] <= median_val]['breakfast_basket_usd']
group2 = df[df['price_usd'] > median_val]['breakfast_basket_usd']

t_stat, p_val = stats.ttest_ind(group1, group2)

print("T-test p-value:", p_val)

if p_val < 0.05:
    print("Reject H0 → Significant difference exists")
else:
    print("Fail to reject H0 → No significant difference")

# -------------------------------
# OBJECTIVE 5: SIMPLE LINEAR REGRESSION
# -------------------------------
print("\n===== SIMPLE LINEAR REGRESSION =====")

X = df[['price_usd']]
y = df['breakfast_basket_usd']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# OBJECTIVE 6: SCENARIO-BASED ANALYSIS
# -------------------------------
print("\n===== SCENARIO ANALYSIS =====")

# Cost per unit
df['cost_per_unit'] = df['breakfast_basket_usd'] / df['quantity']

# Top expensive items
top_items = df.groupby('item')['breakfast_basket_usd'].mean().sort_values(ascending=False).head(5)
print("\nTop Expensive Items:\n", top_items)

# Most expensive countries
top_countries = df.groupby('country')['breakfast_basket_usd'].mean().sort_values(ascending=False).head(5)
print("\nMost Expensive Countries:\n", top_countries)
