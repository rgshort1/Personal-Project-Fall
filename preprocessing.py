#%%
# Import necessary libraries
import pandas as pd

# Load the dataset
data = pd.read_csv("lending_club_loan_two.csv")

# Handle the 'term' column
data['term'] = data['term'].str.replace(' months', '').str.strip().astype(float)

# Handle the 'revol_util' column
data['revol_util'] = (
    data['revol_util']
    .astype(str)
    .str.replace('%', '', regex=True)
    .replace('nan', None)
    .astype(float)
) / 100

# Handle the 'emp_length' column
data['emp_length'] = (
    data['emp_length']
    .str.extract(r'(\d+)')  # Extract numerical part
    .replace('nan', None)
    .astype(float)
)

# Save the cleaned data
cleaned_file_path = "cleaned_data.csv"
data.to_csv(cleaned_file_path, index=False)

print(f"Data cleaned and saved to {cleaned_file_path}")

# %%
