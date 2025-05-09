import pandas as pd
import numpy as np

# Import Scikit Learn
from sklearn.model_selection import train_test_split
import sklearn.linear_model as models
from sklearn.metrics import accuracy_score

# Import Matplot
import matplotlib.pyplot as plt

# Create A Dataframe of the Bank Loans Data From The CSV
bankLoansData = pd.read_csv('Bank Loans.csv')

print(bankLoansData.head())

# Split The Data Into Training Data And Testing Data
xTrain, xTest, yTrain, yTest = train_test_split(bankLoansData[['balance']], bankLoansData['default'], test_size=.8, shuffle=True, random_state=42)

# Train The Logistic Regression Model
model = models.LogisticRegression().fit(xTrain, yTrain)

# Show Models Coefficients
print(f"Model Coefficients: {model.coef_[0][0]}")
print(f"Model Oddsratio: {np.exp(model.coef_[0][0])}")

# Y Prediction
loanPred = model.predict(xTest)


print(model.predict_proba(xTest))

# Evaluate The Model
print(f"Model Accuracy: {accuracy_score(yTest, loanPred):.2f}")

# Allow the user to input a balance amount and predict if the loan will leave to a default
while True:
    try:
        amount = float(input("Input account balance amount to predict if the loan will leave to a default: "))
        prediction = model.predict([[amount]])
        probability = model.predict_proba([[amount]])
        
        print(f"Model Prediction: {'None Default' if prediction[0] == 1 else 'Default'}")
        print(f"Approval probability: {probability[0][1]:.2f}")
    except ValueError:
        print("Please enter a valid number")
    except KeyboardInterrupt:
        print("\nExiting program")
        break
