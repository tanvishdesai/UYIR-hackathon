import pandas as pd

# Load the CSV file
df = pd.read_csv(r'downsized_v2.csv')

# Extract unique states and weather conditions
unique_states = df['State'].value_counts()
unique_weather_conditions = df['Weather_Condition'].value_counts()

# Print the results
print("Unique states in training data:", unique_states)
print("Unique weather conditions:", unique_weather_conditions)