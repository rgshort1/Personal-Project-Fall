# Personal-Project-Fall
# %%
# Credit Default Prediction using Machine Learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('cleaned_data.csv')  # Replace with your actual file name

# Step 2: Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(exclude=['object']).columns

# Step 3: Handle missing values:
# For numeric columns, use mean imputation
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# For categorical columns, use most frequent imputation
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Step 4: Label Encoding for categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Step 5: Separate features (X) and target (y)
X = df.drop(columns=['loan_status'])  # Replace 'loan_status' with your actual target column
y = df['loan_status']  # Replace 'loan_status' with your actual target column

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 8: Standardize features (important for many models)
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

# Step 9: Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Step 10: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# %%
import pandas as pd
import numpy as np
import tkinter as tk

# Example of how to take user input, adjust according to your features and model
def predict_default(model):
    print("Please provide the following information:")

    # Collecting user input (example with common features; replace with your own)
    user_input = {}
    user_input['age'] = float(input("Enter age: "))
    user_input['debt_to_income'] = float(input("Enter debt-to-income ratio: "))
    user_input['credit_score'] = float(input("Enter credit score: "))
    user_input['years_employed'] = float(input("Enter years employed: "))
    user_input['income'] = float(input("Enter income: "))
    user_input['loan_amount'] = float(input("Enter loan amount: "))
    user_input['term'] = input("Enter loan term (short/long): ").lower()
    user_input['home_ownership'] = input("Enter home ownership (rent/mortgage/own): ").lower()
    user_input['purpose'] = input("Enter loan purpose: ").lower()
    user_input['delinq_2yrs'] = float(input("Enter number of delinquencies in the past 2 years: "))
    user_input['revol_util'] = float(input("Enter revolving line utilization rate: "))
    user_input['total_acc'] = float(input("Enter total number of credit lines: "))
    user_input['interest_rate'] = float(input("Enter interest rate: "))

    # Convert the user input into a DataFrame with the correct shape
    user_df = pd.DataFrame([user_input])

    # Preprocess the data if needed (same preprocessing steps you used before training)
    # For example: user_df = preprocess_user_input(user_df)  # Add if you have preprocessing steps

    # Get the probability of default (class 1)
    prediction_proba = model.predict_proba(user_df)

    # The probability of default is the probability of class 1
    probability_of_default = prediction_proba[0][1]  # Probability for class 1 (default)

    # Output the result as a percentage
    print(f"Probability of Default: {probability_of_default * 100:.2f}%")

# Example usage:
# Call predict_default with your trained model (clf is your model here)
predict_default(model)

# User interface for the model inputs
root = tk.Tk()
root.title("Credit Default Prediction")
root.geometry("400x400")

# Create input fields for user to enter data
age_label = tk.Label(root, text="Age:")
age_label.pack()
age_entry = tk.Entry(root)
age_entry.pack()

debt_to_income_label = tk.Label(root, text="Debt-to-Income Ratio:")
debt_to_income_label.pack()
debt_to_income_entry = tk.Entry(root)
debt_to_income_entry.pack()

# Add more input fields for other features...

credit_score_label = tk.Label(root, text="Credit Score:")
credit_score_label.pack()
credit_score_entry = tk.Entry(root)
credit_score_entry.pack()

years_employed_label = tk.Label(root, text="Years Employed:")
years_employed_label.pack()
years_employed_entry = tk.Entry(root)
years_employed_entry.pack()

income_label = tk.Label(root, text="Income:")
income_label.pack()
income_entry = tk.Entry(root)
income_entry.pack()

loan_amount_label = tk.Label(root, text="Loan Amount:")
loan_amount_label.pack()
loan_amount_entry = tk.Entry(root)
loan_amount_entry.pack()

term_label = tk.Label(root, text="Loan Term (short/long):")
term_label.pack()
term_entry = tk.Entry(root)
term_entry.pack()

home_ownership_label = tk.Label(root, text="Home Ownership (rent/mortgage/own):")
home_ownership_label.pack()
home_ownership_entry = tk.Entry(root)
home_ownership_entry.pack()

purpose_label = tk.Label(root, text="Loan Purpose:")
purpose_label.pack()
purpose_entry = tk.Entry(root)
purpose_entry.pack()

delinq_2yrs_label = tk.Label(root, text="Number of Delinquencies in the Past 2 Years:")
delinq_2yrs_label.pack()
delinq_2yrs_entry = tk.Entry(root)
delinq_2yrs_entry.pack()

revol_util_label = tk.Label(root, text="Revolving Line Utilization Rate:")
revol_util_label.pack()
revol_util_entry = tk.Entry(root)
revol_util_entry.pack()

total_acc_label = tk.Label(root, text="Total Number of Credit Lines:")
total_acc_label.pack()
total_acc_entry = tk.Entry(root)
total_acc_entry.pack()

interest_rate_label = tk.Label(root, text="Interest Rate:")
interest_rate_label.pack()
interest_rate_entry = tk.Entry(root)
interest_rate_entry.pack()

# Create a button to trigger the prediction
predict_button = tk.Button(root, text="Predict Default", command=lambda: predict_default(model))
predict_button.pack()

root.mainloop()

# %%
