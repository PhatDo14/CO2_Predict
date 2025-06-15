import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error ,r2_score


def create_ts_data(data, window_size=5):
    i = 1
    while i < window_size:
        data["CO2_{}".format(i)] = data["CO2"].shift(-i)
        i += 1
    data["target"] = data["CO2"].shift(-i)
    data = data.dropna(axis=0)
    return data


data = pd.read_csv("data/co2.csv")

# change Type object to Type DateTime
data["Date"] = pd.to_datetime(data["Date"])
# fill missing value
data["CO2"] = data["CO2"].interpolate()

# Transform date time data to data which ML can work with
data = create_ts_data(data)

# fig, ax = plt.subplots()
# ax.plot(data["Date"], data["CO2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("Co2")
# plt.show()

# Split data
x = data.drop(["Date", "target"], axis=1)
y = data["target"]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

# Train model
reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

# in ra performance cua model
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MsE: {}".format(mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))

# Visualization
# fig, ax = plt.subplots()
# ax.plot(data["Date"][:int(num_samples * train_ratio)], data["CO2"][:int(num_samples * train_ratio)], label="train")
# ax.plot(data["Date"][int(num_samples * train_ratio):], data["CO2"][int(num_samples * train_ratio):], label="test")
# ax.plot(data["Date"][int(num_samples * train_ratio):], y_predict, label="predict")
# ax.set_xlabel("Time")
# ax.set_ylabel("Co2")
# ax.legend()
# ax.grid()
# plt.show()


# deployment
current_data = [380.5, 390, 390.2, 390.4, 393]
# prediction = reg.predict([current_data])[0]
# print(prediction)

for i in range(10):
    print("Input is {}".format(current_data))
    prediction = reg.predict([current_data])[0]
    print("CO2 in week{} is {}".format(i+1, prediction))
    current_data = current_data[1:] + [prediction]
    print("--------------")