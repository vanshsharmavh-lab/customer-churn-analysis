import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("../images", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/customer_churn.csv")

# Basic information
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nDataset Info:")
print(df.info())

# Missing values and duplicates
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Data cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nUpdated Shape:", df.shape)

# Save cleaned dataset
df.to_csv("cleaned_customer_churn.csv", index=False)

# Churn statistics
print("\nChurn Count")
print(df["Churn"].value_counts())

print("\nChurn Percentage")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

print("\nContract Types")
print(df["Contract"].value_counts())

print("\nPayment Methods")
print(df["PaymentMethod"].value_counts())

print("\nInternet Services")
print(df["InternetService"].value_counts())

print("\nSummary Statistics")
print(df.describe())

# Chart 1 - Churn Distribution
df["Churn"].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("../images/churn_distribution.png")
plt.show()

# Chart 2 - Contract Type vs Churn
pd.crosstab(df["Contract"], df["Churn"]).plot(kind="bar")
plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("../images/contract_type.png")
plt.show()

# Chart 3 - Payment Method vs Churn
pd.crosstab(df["PaymentMethod"], df["Churn"]).plot(kind="bar")
plt.title("Payment Method vs Churn")
plt.xlabel("Payment Method")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("../images/payment_method.png")
plt.show()

# Chart 4 - Monthly Charges
plt.hist(df["MonthlyCharges"], bins=30)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("../images/monthly_charges.png")
plt.show()

# Business insights
print("\nOverall Churn Rate:")
print(round((df["Churn"] == "Yes").mean() * 100, 2), "%")

print("\nAverage Monthly Charges")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

print("\nAverage Customer Tenure")
print(df.groupby("Churn")["tenure"].mean().round(2))

print("\nContract Type vs Churn")
print(df.groupby("Contract")["Churn"].value_counts())

print("\nPayment Method vs Churn")
print(df.groupby("PaymentMethod")["Churn"].value_counts())

print("\nAnalysis completed successfully.")
