import datetime as dt
import json
import os
from urllib import request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def get_data(ticker):
    api_key = "Your API Key"

    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"

    file_to_save = f'stock_market_data-{ticker}.csv'

    if not os.path.exists(file_to_save):
        with request.urlopen(url_string) as url:

            data = json.loads(url.read().decode())
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                            float(v['4. close']), float(v['1. open'])]
                df.loc[-1, :] = data_row
                df.index = df.index + 1
        print('Data saved to : %s' % file_to_save)
        df.to_csv(file_to_save)

    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)
    return df


if __name__ == '__main__':
    dd = get_data("AAPL")
    dd = dd.sort_values('Date')
    dd['Date'] = pd.to_datetime(dd['Date'])
    nd = dd[['Date', 'Close']]

    scaler = MinMaxScaler()
    nd.index = nd.Date
    nd.drop("Date", axis=1, inplace=True)
    fd = nd.values
    td = fd[:len(fd) // 10 * 8]
    vd = fd[len(fd) // 10 * 8:]

    prediction_days = 60
    scaled_data = scaler.fit_transform(fd)
    x_train_data, y_train_data = [], []
    for i in range(prediction_days, len(td)):
        x_train_data.append(scaled_data[i - prediction_days:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data, epochs=15, batch_size=20)

    lstm_model.save('lstm_model.h5')
    inputs_data = nd[len(nd) - len(vd) - prediction_days:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(prediction_days, inputs_data.shape[0]):
        X_test.append(inputs_data[i - prediction_days:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_closing_price = lstm_model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
    valid_data = nd[len(fd) // 10 * 8:]
    valid_data['Predicted'] = predicted_closing_price

    plt.plot(valid_data[['Close', 'Predicted']])
    plt.xticks(rotation=45)

    plt.show()
