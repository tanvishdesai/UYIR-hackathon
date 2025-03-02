import pandas as pd

# Load the CSV file into a DataFrame
file_path = r"C:\Users\DELL\Downloads\list_eval_partition.csv\list_eval_partition.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Print the first three rows
# Select the first 3 rows of the DataFrame
first_three_rows = df.head(3)

# Save these rows to a new CSV file
output_file_path = r"C:\Users\DELL\Downloads\list_eval_partition.csv\eval1.csv"  # Replace with your desired output file path
first_three_rows.to_csv(output_file_path, index=False)  # Set index=False to avoid saving row indices

print(f"First 3 rows saved to {output_file_path}")