# ==========================================
# 🚗 Car Price Prediction using Random Forest
# ==========================================

# Import required libraries
import pandas as pd

# ------------------------------------------
# 📂 Load Dataset from Local System
# ------------------------------------------
from google.colab import files
uploaded = files.upload()

# Read the dataset
dataset = pd.read_csv('dataset.csv')

# Drop unnecessary column (ID column not useful for prediction)
dataset = dataset.drop(['car_ID'], axis=1)

# ------------------------------------------
# 📊 Dataset Overview
# ------------------------------------------
print("Dataset Shape:", dataset.shape)
print("\nFirst 5 Rows:\n", dataset.head(5))

# ------------------------------------------
# 🎯 Split Dataset into Features (X) and Target (Y)
# ------------------------------------------

# Independent variables (features)
Xdata = dataset.drop('price', axis=1)

# Select only numerical columns (ignore text/categorical data)
numericalCols = Xdata.select_dtypes(exclude=['object']).columns
X = Xdata[numericalCols]

# Dependent variable (target)
Y = dataset['price']

# ------------------------------------------
# 📏 Feature Scaling
# ------------------------------------------
from sklearn.preprocessing import scale

# Store column names before scaling
cols = X.columns

# Apply scaling (normalization)
X = pd.DataFrame(scale(X), columns=cols)

# ------------------------------------------
# 🔀 Train-Test Split
# ------------------------------------------
from sklearn.model_selection import train_test_split

# Split data into training (80%) and testing (20%)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0
)

# ------------------------------------------
# 🌲 Train Random Forest Model
# ------------------------------------------
from sklearn.ensemble import RandomForestRegressor

# Initialize model
model = RandomForestRegressor()

# Train the model
model.fit(x_train, y_train)

# ------------------------------------------
# 📈 Model Evaluation
# ------------------------------------------

# Predict on test data
ypred = model.predict(x_test)

# Calculate R2 Score
from sklearn.metrics import r2_score

r2score = r2_score(y_test, ypred)

# Print accuracy
print("\nModel Performance:")
print("R2 Score: {:.2f}%".format(r2score * 100))