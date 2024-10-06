import streamlit as st
import pytesseract
from PIL import Image
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

# Function to extract text from an uploaded image (if payslip is an image)
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Parse text to extract financial details
def parse_payslip(text):
    # Define regex patterns for extracting key fields
    income_pattern = r'(?i)(gross salary|net income|total earnings):?\s*\$?([0-9,]+)'
    expense_pattern = r'(?i)(deductions|total deductions|expenses):?\s*\$?([0-9,]+)'
    credit_score_pattern = r'(?i)(credit score):?\s*([0-9]+)'

    # Initialize extracted values
    monthly_income = 0
    monthly_expenses = 0
    credit_score = 0

    # Extract monthly income (e.g., from Gross Salary or Net Income)
    income_match = re.search(income_pattern, text)
    if income_match:
        # Convert to a numerical value, remove commas if any
        monthly_income = float(income_match.group(2).replace(',', ''))

    # Extract monthly expenses (e.g., from Deductions)
    expense_match = re.search(expense_pattern, text)
    if expense_match:
        # Convert to a numerical value, remove commas if any
        monthly_expenses = float(expense_match.group(2).replace(',', ''))

    # Extract credit score (if available)
    credit_score_match = re.search(credit_score_pattern, text)
    if credit_score_match:
        credit_score = int(credit_score_match.group(2))

    # Return the parsed values
    return monthly_income, monthly_expenses, credit_score

# Load and preprocess bank statement data (dummy data in this case)
def load_data():
    data = {
        'Monthly_Income': [3000, 4500, 5000, 3500, 4000, 3800, 3200],
        'Monthly_Expenses': [2000, 2500, 2700, 2300, 2200, 2100, 1900],
        'Loan_Amount': [5000, 10000, 12000, 8000, 7000, 6000, 6500],
        'Loan_Term': [24, 36, 48, 24, 36, 24, 36],
        'Credit_Score': [650, 700, 710, 670, 680, 690, 660],
        'Loan_Approved': [1, 1, 1, 0, 0, 0, 1]  # 1 = Approved, 0 = Not Approved
    }
    df = pd.DataFrame(data)
    return df

# Preprocess the data and split it into training and testing sets
def preprocess_data(df):
    X = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Train a machine learning model
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Main application function
def main():
    st.title("Loan Underwriting Application Using Payslip")

    st.write("This application uses payslips to assess loan eligibility based on income, expenses, and credit score.")

    # Allow user to upload a payslip file (PDF or image)
    uploaded_file = st.file_uploader("Upload a payslip (PDF or image)", type=['pdf', 'png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if 'pdf' in file_type:
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_image(uploaded_file)

        st.write("Extracted Text from Payslip:")
        st.write(text)

        # Parse the payslip text to extract financial details
        monthly_income, monthly_expenses, credit_score = parse_payslip(text)

        st.write(f"Parsed Monthly Income: {monthly_income}")
        st.write(f"Parsed Monthly Expenses: {monthly_expenses}")
        st.write(f"Parsed Credit Score: {credit_score}")

        # Load and preprocess data
        data = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(data)

        # Train the model
        model = train_model(X_train, y_train)

        # Loan input features from parsed payslip
        loan_amount = st.number_input("Loan Amount", min_value=1000, value=5000)
        loan_term = st.slider("Loan Term (months)", min_value=12, max_value=60, value=24)

        # Predict Loan Approval
        if st.button("Predict Loan Approval"):
            input_data = np.array([[monthly_income, monthly_expenses, loan_amount, loan_term, credit_score]])
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.success("Loan Approved!")
            else:
                st.error("Loan Not Approved!")

        # Display model accuracy (using dummy data)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
