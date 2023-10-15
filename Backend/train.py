import pandas as pd

# Load your CSV file
csv_file_path = '/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/_annotations.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file_path)

# Examine the column names in your DataFrame
print("Column names in your CSV file:")
print(data.columns)

# Once you've identified the correct column name, replace 'your_column_name' below
correct_column_name = 'class'  # Replace with the actual column name containing class labels

# Create a mapping of unique class labels to numerical values
label_mapping = {label: idx for idx, label in enumerate(data[correct_column_name].unique())}

# Apply label encoding to the DataFrame
data['encoded_labels'] = data[correct_column_name].map(label_mapping)

# Save the updated data to a new CSV file
updated_csv_file_path = '/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/_annotations.csv'  # Replace with the path where you want to save the updated CSV
data.to_csv(updated_csv_file_path, index=False)

# Resulting DataFrame with encoded labels
print(data)

# Check the new CSV file for updated data
print(f"Updated CSV file saved to {updated_csv_file_path}")
