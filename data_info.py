import pandas as pd

# Load the CSV file into a DataFrame
file_path = r'us-accidents\US_Accidents_March23.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Calculate the percentage of empty values for each column
empty_percentage = (df.isnull().mean() * 100).round(2)

# Display the result
print("Percentage of empty values for each column:")
print(empty_percentage)