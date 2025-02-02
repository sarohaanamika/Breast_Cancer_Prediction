import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("C:/Users/aksha/Desktop/ML_Pr/Project/Ujala_Cygnus_breast-cancer_updated.csv")

# Load the data (assuming it's a CSV file)
df = data

# Separate the 'id' and 'diagnosis' columns
ids = df['id']
diagnosis = df['diagnosis']

# Drop 'id' and 'diagnosis' columns for scaling
features = df.drop(['id', 'diagnosis'], axis=1)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply Min-Max Scaling to the features
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Add the 'id' and 'diagnosis' columns back to the scaled DataFrame
scaled_df['id'] = ids
scaled_df['diagnosis'] = diagnosis

# Reorder the columns so 'id' and 'diagnosis' come first
scaled_df = scaled_df[['id', 'diagnosis'] + [col for col in scaled_df.columns if col not in ['id', 'diagnosis']]]

# Save the scaled DataFrame to a new CSV file
scaled_df.to_csv('scaled_breast_cancer_data.csv', index=False)

print("Min-Max Scaling applied and data saved to 'scaled_breast_cancer_data.csv'")