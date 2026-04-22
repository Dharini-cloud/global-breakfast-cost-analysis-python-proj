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
# LOAD DATA
# -------------------------------
df = pd.read_csv("breakfast basket.csv")

# -------------------------------
# DATA CLEANING
# -------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df.dropna(how='all')

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# Fill missing values
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert date column safely
df['data_collection_date'] = pd.to_datetime(df['data_collection_date'], errors='coerce')

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Remove constant columns
nunique = df[num_cols].nunique()
constant_cols = nunique[nunique <= 1].index
df = df.drop(columns=constant_cols)

# Update numeric columns
num_cols = df.select_dtypes(include=np.number).columns

# -------------------------------
#  EDA 
# -------------------------------
print("\n===== EDA =====")
print("Shape:", df.shape)
print("\nInfo:\n")
print(df.info())
print("\nSummary:\n", df.describe())
print("\nSkewness:\n", df.skew(numeric_only=True))

# -------------------------------
# OBJECTIVE 1: VISUALIZATION
# -------------------------------
sns.set(style="whitegrid")

# 1 Histogram
plt.figure()
sns.histplot(df['breakfast_basket_usd'], kde=True)
plt.title("Distribution of Breakfast Basket Cost")
plt.show()

# 2 Boxplot
plt.figure()
sns.boxplot(x=df['breakfast_basket_usd'])
plt.title("Outlier Detection")
plt.show()

# 3 Bar Chart
plt.figure()
top_items = df.groupby('item')['breakfast_basket_usd'].mean().sort_values(ascending=False).head(10)
top_items.plot(kind='bar')
plt.title("Top 10 Items by Cost")
plt.xticks(rotation=45)
plt.show()

# 4 Scatter Plot
plt.figure()
sns.scatterplot(x='price_usd', y='breakfast_basket_usd', data=df)
plt.title("Price vs Basket Cost")
plt.show()

# 5 Heatmap
plt.figure()
important_cols = ['price_usd','exchange_rate','yoy_inflation_estimate_pct','population_estimate','breakfast_basket_usd']
sns.heatmap(df[important_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 6 Pairplot (reduced columns for clarity)
sns.pairplot(df[['price_usd', 'exchange_rate', 'population_estimate', 'breakfast_basket_usd']])
plt.show()

# 7 Pie Chart
plt.figure()
pie_data = df['item'].value_counts().head(5)
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
plt.title("Top 5 Items Distribution")
plt.show()

# 8 Line Chart (trend over time)
df_sorted = df.sort_values(by='data_collection_date')

plt.figure()
plt.plot(df_sorted['data_collection_date'], df_sorted['breakfast_basket_usd'])
plt.xticks(rotation=45)
plt.title("Trend of Breakfast Basket Cost Over Time")
plt.tight_layout()
plt.show()

# -------------------------------
# OBJECTIVE 2: OUTLIER DETECTION
# -------------------------------
print("\n===== OUTLIER DETECTION =====")

num_df = df[num_cols]

# IQR Method
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

# Z-score Method
z_scores = np.abs(zscore(num_df, nan_policy='omit'))
df_z = num_df[(z_scores < 3).all(axis=1)]

print("\nCorrelation After Z-score:\n", df_z.corr())

# -------------------------------
# OBJECTIVE 3: HYPOTHESIS TESTING
# -------------------------------
print("\n===== HYPOTHESIS TESTING =====")

sample = df['price_usd'].sample(500, random_state=42)
stat, p = stats.shapiro(sample)
print("Shapiro p-value:", p)

median_val = df['price_usd'].median()

g1 = df[df['price_usd'] <= median_val]['breakfast_basket_usd']
g2 = df[df['price_usd'] > median_val]['breakfast_basket_usd']

t_stat, p_val = stats.ttest_ind(g1, g2)
print("T-test p-value:", p_val)

# -------------------------------
# OBJECTIVE 4: LINEAR REGRESSION
# -------------------------------
print("\n===== LINEAR REGRESSION =====")

X = df[['price_usd']]
y = df['breakfast_basket_usd']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# OBJECTIVE 5: SCENARIO ANALYSIS
# -------------------------------
print("\n===== SCENARIO ANALYSIS =====")

df['cost_per_unit'] = df['breakfast_basket_usd'] / df['quantity']

top_items = df.groupby('item')['breakfast_basket_usd'].mean().sort_values(ascending=False).head(5)
print("\nTop Expensive Items:\n", top_items)

top_countries = df.groupby('country')['breakfast_basket_usd'].mean().sort_values(ascending=False).head(5)
print("\nMost Expensive Countries:\n", top_countries)
