import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# read in dataset
file_path = "apple_sales_2024.csv"  
data = pd.read_csv(file_path)

# normalization
scaler = MinMaxScaler()
numerical_columns = [
    "iPhone Sales (in million units)",
    "iPad Sales (in million units)",
    "Mac Sales (in million units)",
    "Wearables (in million units)",
    "Services Revenue (in billion $)"
]
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# OHE
encoder = OneHotEncoder(sparse_output=False, drop="first")
categorical_columns = ["State", "Region"] 
encoded_data = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)

# Combine normalized + encoded data
processed_data = pd.concat([data[numerical_columns], encoded_df], axis=1)

# store preprocessed data
processed_data.to_csv("preprocessed_apple_sales.csv", index=False)

print("Preprocessing completed. Data saved as 'preprocessed_apple_sales.csv'")