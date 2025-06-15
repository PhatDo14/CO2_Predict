import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error ,r2_score


def create_ts_data(data, window_size=5, target_size=3):
    i = 1
    while i < window_size:
        data["CO2_{}".format(i)] = data["CO2"].shift(-i)
        i += 1
    i=0
    while i < target_size:
        data["target_{}".format(i+1)] = data["CO2"].shift(-i-window_size)
        i += 1
    data = data.dropna(axis=0)
    return data

data = pd.read_csv("data/co2.csv")

# change Type object to Type DateTime
data["Date"] = pd.to_datetime(data["Date"])
# fill missing value
data["CO2"] = data["CO2"].interpolate()

# Transform date time data to data which ML can work with
window_size = 5
target_size = 3
data = create_ts_data(data, window_size, target_size)

# Split data
targets = ["target_{}".format(i+1) for i in range(target_size)]
x = data.drop(["Date"] + targets, axis=1)
y = data[targets]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

#Create 3 model for 3 target
regs = [LinearRegression() for _ in range(target_size)]
for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i+1)])

r2 = []
mse = []
mae = []
for i, reg in enumerate(regs):
    y_predict = reg.predict(x_test)
    mae.append(mean_absolute_error(y_test["target_{}".format(i+1)], y_predict))
    mse.append(mean_squared_error(y_test["target_{}".format(i+1)], y_predict))
    r2.append(r2_score(y_test["target_{}".format(i+1)], y_predict))

# print("R2: {}".format(r2))
# print("MSE: {}".format(mse))
# print("MAE: {}",format(mae))

#deployment
current_data = [380.5, 390, 390.2, 390.4, 393]
# prediction = reg.predict([current_data])[0]
# print(prediction)

for i in range(4):
    prediction = []
    for j,reg in enumerate(regs):
        print("Input is {}".format(current_data))
        prediction.append(reg.predict([current_data])[0])
        print("CO2 in week{} is {}".format(j+6, prediction[j]))
        print("--------------")
    current_data = current_data[3:] + prediction
    print("************")
