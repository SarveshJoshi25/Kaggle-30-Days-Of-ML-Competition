import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Reading Data, Storing into different variables
train_X = pd.read_csv("/home/sarvesh/PycharmProjects/Machine-Learning/30-days-of-ML/Datasets/train.csv")
test_X = pd.read_csv("/home/sarvesh/PycharmProjects/Machine-Learning/30-days-of-ML/Datasets/test.csv")

train_id = train_X["id"]
test_id = test_X["id"]
train_y = train_X["target"]
train_X.drop(["id", "target"], axis=1, inplace=True)
test_X.drop(["id"], axis=1, inplace=True)

# Initialised some objects
labelEncoder = LabelEncoder()
min_max_scaler = MinMaxScaler()
linear_regression = LinearRegression()

# Label Encoding
for cols in train_X.columns:
    if train_X[cols].dtype == "O":
        train_X[cols] = labelEncoder.fit_transform(train_X[cols])
        test_X[cols] = labelEncoder.fit_transform(test_X[cols])

# Normalised the Data
train_X_values = train_X.values
scaled = min_max_scaler.fit_transform(train_X_values)
train_X = pd.DataFrame(scaled)

test_X_values = test_X.values
scaled = min_max_scaler.fit_transform(test_X_values)
test_X = pd.DataFrame(scaled)

y_min = min(train_y)
y_max = max(train_y)
for i in range(len(train_y)):
    train_y[i] = (train_y[i] - y_min) / (y_max - y_min)

# Prediction and Storing
lr = linear_regression.fit(train_X, train_y)
predictions = lr.predict(test_X)
submission = [["id", "target"]]

for i in range(len(predictions)):
    prediction = (predictions[i] * (y_max - y_min)) + y_min
    submission.append([test_id[i], prediction])

with open("/home/sarvesh/PycharmProjects/Machine-Learning/30-days-of-ML/Submissions/submission1.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(submission)
