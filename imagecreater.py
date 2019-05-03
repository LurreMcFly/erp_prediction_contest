from pyts.image import GADF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
date = open("date.txt", "r").read()

def create_percentage(df):
    df["percentage"] = df["CLOSE"].pct_change()


def create_RSI(df, days=14):
    delta = df["CLOSE"].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up = up.rolling(days).mean()
    roll_down = down.rolling(days).mean().abs()

    RS = roll_up / roll_down

    df['RSI'] = 100 - (100 / (1 + RS))

def create_MACD(df, n_slow = 26, n_fast = 10):
    emaslow = df["CLOSE"].ewm(span=n_slow, min_periods=1).mean()
    emafast = df["CLOSE"].ewm(span=n_fast, min_periods=1).mean()
    df["MACD"] = emafast - emaslow

def create_bollinger_bands(df, days=30):
    ave = df["CLOSE"].rolling(days).mean()
    sd = df["CLOSE"].rolling(days).std()
    df["BBU"] = ave + (sd*2)
    df["BBL"] = ave - (sd*2)
    df["BBdiff"] = (df["CLOSE"]-((df["BBU"]+df["BBL"])/2))/(df["BBU"]-df["BBL"])

def create_moving_average(df, days1 = 200, days2 = 50):
    df["MA200"] = df["CLOSE"].rolling(days1).mean()
    df["MA200diff"] = df["CLOSE"] - df["MA200"]
    df["MA50"] = df["CLOSE"].rolling(days2).mean()
    df["MA50diff"] = df["CLOSE"] - df["MA50"]
    df["MA20050diff"] = df["MA50"] - df["MA200"]


def add_features(df):
    retun_df = df.copy()

    create_percentage(retun_df)
    create_RSI(retun_df)
    create_MACD(retun_df)
    create_bollinger_bands(retun_df)
    create_moving_average(retun_df)

    return retun_df

new_df = add_features(df)
features = ['CLOSE', 'percentage', 'RSI', 'MACD', 'BBdiff', 'MA20050diff']

image_size = 64
n = len(df)

def create_gadf(df, idx, columns, image_size, date, label, j):
    X = np.transpose(df[columns][(idx - image_size):idx].values)
    gadf = GADF(image_size)
    X_gadf = gadf.fit_transform(X)
    variable = columns[j]

    if (date.split("-")[0] == '2017'):
        location = "images/test_" + variable + "/"
    elif (int(date.split("-")[0]) % 11 == 0):
        location = "images/validation_" + variable+ "/"
    else:
        location = "images/train_" + variable + "/"

    dir = location + date + "," + label + ".png"
    plt.imsave(dir, X_gadf[j], cmap='rainbow', origin='lower')

start_date = '1960-01-15'
start_idx = df.index[df.iloc[:,0] == start_date][0]

for j in range(len(features)):
    for i in range(start_idx+image_size, n):
        date = str(new_df["Unnamed: 0"][i])
        label = str(new_df["ASPFWR5"][i])
        create_gadf(new_df, i, features, image_size, date, label, j)
        if (i%300 == 0):
            print(date.split("-")[0]+" "+features[j])
