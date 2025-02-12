import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "processed_with_target.csv"
data = pd.read_csv(file_path)

# just prints headers and first couple rows
print(data.info())
print(data.head())

# plot the distribution of the most popular device
plt.figure(figsize=(8,5))
sns.countplot(x=data["most_popular_device"], palette="Set2")
plt.title("Distribution of Most Popular Devices")
plt.xlabel("Device Type")
plt.ylabel("Count")
plt.show()

missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

""" Compute correlation matrix (excluding the target column)
correlation_matrix = data.drop(columns=["most_popular_device"]).corr()


plt.figure(figsize=(12,10))


sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
            annot_kws={"size": 8}, mask=abs(correlation_matrix) < 0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

# Title
plt.title("Feature Correlation Matrix", fontsize=14)

# Show the plot
plt.show()
"""
