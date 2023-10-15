import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/train/_annotations.csv')

# Sort the DataFrame by the 'class' column
sorted_df = df.sort_values(by='class')

# Save the sorted DataFrame back to a CSV file
sorted_df.to_csv('/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/train/_annotations.csv', index=False)
