import pandas as pd

file_path = "preprocessed_apple_sales.csv"  
processed_data = pd.read_csv(file_path)

device_columns = [
    "iPhone Sales (in million units)",
    "iPad Sales (in million units)",
    "Mac Sales (in million units)",
    "Wearables (in million units)"
]

# getting most popular device in each row of data
most_popular_colnames = processed_data[device_columns].idxmax(axis=1)

column_to_label_map = {
    "iPhone Sales (in million units)": "iPhone",
    "iPad Sales (in million units)": "iPad",
    "Mac Sales (in million units)": "Mac",
    "Wearables (in million units)": "Wearables"
}
processed_data["most_popular_device"] = most_popular_colnames.map(column_to_label_map)

print("Distribution of Most Popular Device:")
print(processed_data["most_popular_device"].value_counts())

processed_data.to_csv("processed_with_target.csv", index=False)

print("data saved as 'processed_with_target.csv'.")
