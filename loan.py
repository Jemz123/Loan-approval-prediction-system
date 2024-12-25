import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset and strip whitespace from column names
data = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\loan.csv")
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces

# Display the column names to verify
print("Column Names in the Dataset:")
print(data.columns)

# Check if the target column exists
TARGET_COLUMN = 'loan_status'  # Target column
if TARGET_COLUMN not in data.columns:
    raise ValueError(f"The dataset does not contain a '{TARGET_COLUMN}' column.")

# Data preprocessing
# Fill missing values
for column in data.columns:
    if data[column].dtype == 'object':  # Categorical
        data[column] = data[column].fillna(data[column].mode()[0])
    else:  # Numerical
        data[column] = data[column].fillna(data[column].mean())

# Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Features and target variable
X = data.drop(columns=[TARGET_COLUMN])  # Features
y = data[TARGET_COLUMN]                 # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the model using Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict on a new applicant's data
new_applicant = {
    'loan_id': 100,  # Dummy ID, not used in prediction
    'no_of_dependents': 1,
    'education': 'Graduate',
    'self_employed': 'No',
    'income_annum': 8000000,
    'loan_amount': 15000000,
    'loan_term': 10,
    'cibil_score': 750,
    'residential_assets_value': 2000000,
    'commercial_assets_value': 5000000,
    'luxury_assets_value': 10000000,
    'bank_asset_value': 7000000
}

# Convert the new applicant's data to DataFrame
new_applicant_df = pd.DataFrame([new_applicant])

# Apply the same label encoding to the new applicant data
for column in categorical_columns:
    if column in new_applicant_df.columns:
        try:
            # Transform using the existing label encoder
            new_applicant_df[column] = label_encoder.transform(new_applicant_df[column])
        except ValueError as e:
            # Handle unseen labels by mapping them to a default value (e.g., -1)
            print(f"Unseen label in column '{column}': {e}")
            new_applicant_df[column] = new_applicant_df[column].apply(
                lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
            )

# Align the new applicant's features with the training dataset
missing_cols = set(X.columns) - set(new_applicant_df.columns)
for col in missing_cols:
    new_applicant_df[col] = 0  # Add missing columns with default values
new_applicant_df = new_applicant_df[X.columns]  # Ensure the column order matches

# Make prediction for the new applicant
approval_prediction = model.predict(new_applicant_df)
print(f"\nLoan Approval Status for New Applicant: {'Approved' if approval_prediction[0] == 1 else 'Rejected'}")
