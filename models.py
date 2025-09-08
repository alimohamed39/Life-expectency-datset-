import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from architecture import FNN
from main import *
import tqdm

data = create_data()
data = drop_columns(data)
data = encode(data)


X = data.drop(columns=["Life expectancy "])
y = data["Life expectancy "]

def baseline():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline_pred = np.full_like(y_test, y_test.mean())
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    print("Baseline MSE:", baseline_mse)


def linear_regression():

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)


#    print("The train loss according to the linear regression is: ", mse_train)
    print("The test loss according to the linear regression is: ", mse)
    return mse



def random_forest():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, rf_pred)
    r2 = r2_score(y_test, rf_pred)
    print("The test loss according to the random forest regression is: ", mse)
    print("The accuracy loss according to the random forest regression is: ", r2)

def train_FNN(optimizer,loss=torch.nn.MSELoss,model=FNN,epochs=10):

  #  optimizer(model.parameters(), lr=0.001)
    model = model()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()


        if (epoch + 1) % 10 == 0:
            model.eval()
            predictions = model(X_test_t)
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {predictions:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t)
        mse = loss(y_pred, y_test_t).item()
        print("Final MSE:", mse)
















mse = baseline()
mse = linear_regression()
mse = random_forest()
model = FNN()
optimzer = torch.optim.Adam(model.parameters(), lr=0.001)
mse = train_FNN(optimizer = optimzer)

