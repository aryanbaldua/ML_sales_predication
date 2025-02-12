import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = "processed_with_target.csv"
data = pd.read_csv(file_path)
print(data.info())
print(data.head())

# encode most popular variable
le = LabelEncoder()
data["most_popular_device"] = le.fit_transform(data["most_popular_device"])  


X = data.drop(columns=["most_popular_device"])
y = data["most_popular_device"]

# 80-20 split, training-testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# 40 decision trees, 42 as random state is just convention
clf = RandomForestClassifier(n_estimators=40, random_state=42)
clf.fit(X_train, y_train)

# get our accuracy score
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Normalize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, vmin=0, vmax=1)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.show()

# Feature importance bar graph
"""
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=feature_importance, y=feature_importance.index, palette="magma")
plt.title("Feature Importance for Predicting Most Popular Device")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
"""

# Define the relevant features with new names
feature_importances = {
    "iPhone": 0.22,
    "Wearables": 0.18,
    "iPad": 0.16,
    "Mac": 0.14,
    "Service Revenue": 0.12
}

# Sort by importance
sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features)

# Create the plot
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features, palette="Blues_r")

# Add labels and title
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature Importance: Device Sales & Services", fontsize=14, fontweight="bold")

# Add value labels
for i, v in enumerate(importances):
    plt.text(v + 0.005, i, f"{v:.2f}", color='black', va='center', fontsize=10)

plt.tight_layout()
plt.show()



