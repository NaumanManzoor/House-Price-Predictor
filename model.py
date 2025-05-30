import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Load data
data = pd.read_csv('housing.csv')
data.dropna(inplace=True)

# Features and target
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Define preprocessor for 'ocean_proximity'
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['ocean_proximity'])],
    remainder='passthrough'
)

# Fit preprocessor and transform features
X_processed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save both model and preprocessor in one pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'preprocessor': preprocessor}, f)

print("✅ Model and preprocessor saved successfully as model.pkl")
