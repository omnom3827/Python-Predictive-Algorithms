import pandas as pd
import numpy as np

# Import Scikit Learn
from sklearn.model_selection import train_test_split
import sklearn.linear_model as models
from sklearn.metrics import mean_squared_error, r2_score

#Import Matplot
import matplotlib.pyplot as plt

# Create A Dataframe Of Car CO2 Data From The CSV
salaryData = pd.read_csv('co2.csv')

# Split The Data Into Training Data And Testing Data
xTrain, xTest, yTrain, yTest = train_test_split(salaryData[['Engine Size(L)']], salaryData['CO2 Emissions(g/km)'], test_size=.8)

print(xTest, yTest)
# Train The Linear Regression Model
model = models.LinearRegression().fit(xTrain, yTrain)
# Y Prediction
yPred = model.predict(xTest)


# Show The Mean Squared Error And The Coefficient Of Determination
print(f"Mean squared error: {mean_squared_error(yTest, yPred):.2f}")
print(f"Coefficient of determination: {r2_score(yTest, yPred):.2f}")

# Make A Scatter Plot Of The Training Data
fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].scatter(xTrain, yTrain, label="Train data points")
ax[0].plot(
    xTrain,
    model.predict(xTrain),
    linewidth=3,
    color="tab:orange",
    label="Model predictions",
)
ax[0].set(xlabel="Engine Size (L)", ylabel="CO2 Emissions(g/km)", title="Train set")
ax[0].legend()

ax[1].scatter(xTest, yPred, label="Test data points")
ax[1].plot(xTest, yPred, linewidth=3, color="tab:orange", label="Model predictions")
ax[1].set(xlabel="Engine Size (L)", ylabel="CO2 Emissions(g/km)", title="Test set")
ax[1].legend()

fig.suptitle("Linear Regression")

plt.show()

plt.savefig('engine_size.png')
print("Plot saved as 'salary_plot.png'")

# Allow the user to input a engine size amount and predict the CO2 emissions
while True:
    try:
        amount = float(input("Input engine size amount to predict the CO2 emissions: "))
        prediction = model.predict([[amount]])
        print(f"Model Prediction: {prediction[0]:.2f} g/km")
    except ValueError:
        print("Please enter a valid number")
    except KeyboardInterrupt:
        print("\nExiting program")
        break