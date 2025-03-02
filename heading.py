# Import necessary library
import pandas as pd

# Load the CSV file
def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Failed to load CSV: {e}")

# Print column headings
def print_column_headings(data):
    column_headings = data.columns.tolist()
    for i, heading in enumerate(column_headings):
        print(f"{i+1}. {heading}")

# Main function
def main():
    file_path = r'us-accidents\US_Accidents_March23.csv'  # Replace with your CSV file path
    data = load_csv(file_path)
    if data is not None:
        print_column_headings(data)

if __name__ == "__main__":
    main()