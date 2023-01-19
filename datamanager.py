import pandas as pd
import numpy as np


"""
하루 단위 가격 변화율을 feature로 사용한다.
한 종목에 대한 데이터 예시는 아래 데이터프레임과 같다.
모든 종목에 대한 Tensor shpae은 (L, F, S)이다.
L: Len of data
F: Num of features
S: Num of Stocks 
            open_ratio  high_ratio  low_ratio  close_ratio  volume_ratio  Price
Date                                                                           
2010-01-05    0.005249    0.005222   0.005277     0.002618      0.219087   7.66
2010-01-06    0.000000   -0.001299  -0.011811    -0.016971     -0.082638   7.53
2010-01-07   -0.013055   -0.015605  -0.007968    -0.001328     -0.135885   7.52
2010-01-08   -0.006614    0.000000   0.000000     0.006649     -0.061304   7.57
2010-01-11    0.011984    0.005284  -0.004016    -0.009247      0.032040   7.50
"""

class DataManager:
    Features1 = ["Open", "High", "Low", "Close", "Volume", "Price"]
    Features2 = ["open_ratio", "high_ratio", "low_ratio", "close_ratio", "volume_ratio", "Price"]
    Features3 = ["momentum1", "momentum10", "momentum30", "momentum50", "momentum70", "Price"]
    Features4 = ["close_ratio", "Price"]

    def __init__(self, paths, train_start=None, train_end=None, test_start=None, test_end=None):
        self.paths = paths
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

    def get_data(self, path):

        data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
        data = data.replace(0, np.nan)
        data = data.fillna(method="bfill")
        data.insert(data.shape[1], "Price", data["Close"].values)
        data = self.get_ratio(data)
        data = self.get_momentum(data)
        data = data.dropna()

        train_start = data["Date"].iloc[0] if self.train_start is None else self.train_start
        train_end = data["Date"].iloc[-1] if self.train_end is None else self.train_end
        test_start = data["Date"].iloc[0] if self.test_start is None else self.test_start
        test_end = data["Date"].iloc[-1] if self.test_end is None else self.test_end

        train_data = data[(data["Date"] >= train_start) & (data["Date"] <= train_end)]
        test_data = data[(data["Date"] >= test_start) & (data["Date"] <= test_end)]

        train_data = train_data.set_index("Date", drop=True)
        test_data = test_data.set_index("Date", drop=True)

        train_data = train_data.astype("float32").loc[:,self.Features3]
        test_data = test_data.astype("float32").loc[:,self.Features3]
        return train_data, test_data

    def get_data_tensor(self):

        for path in self.paths:
            train_data, test_data = self.get_data(path)

            if path == self.paths[0]:
                common_date_train = set(train_data.index.unique())
                common_date_test = set(test_data.index.unique())
            else:
                common_date_train = common_date_train & set(train_data.index.unique())
                common_date_test = common_date_test & set(test_data.index.unique())

        common_date_train = list(common_date_train)
        common_date_test = list(common_date_test)
        common_date_train.sort()
        common_date_test.sort()

        for path in self.paths:
            train_data_, test_data_ = self.get_data(path)
            train_data_ = train_data_[train_data_.index.isin(common_date_train)].to_numpy()
            test_data_ = test_data_[test_data_.index.isin(common_date_test)].to_numpy()

            train_data_ = train_data_[:, :, np.newaxis]
            test_data_ = test_data_[:, :, np.newaxis]

            if path == self.paths[0]:
                train_data = train_data_
                test_data = test_data_
            else:
                train_data = np.concatenate([train_data, train_data_], axis=-1)
                test_data = np.concatenate([test_data, test_data_], axis=-1)

        return train_data, test_data

    @staticmethod
    def get_ratio(data):
        day = 1
        data.loc[day:, "open_ratio"] = (data["Open"][day:].values - data["Open"][:-day].values) / data["Open"][:-day].values 
        data.loc[day:, "high_ratio"] = (data["High"][day:].values - data["High"][:-day].values) / data["High"][:-day].values
        data.loc[day:, "low_ratio"] = (data["Low"][day:].values - data["Low"][:-day].values) / data["Low"][:-day].values
        data.loc[day:, "close_ratio"] = (data["Close"][day:].values - data["Close"][:-day].values) / data["Close"][:-day].values
        data.loc[day:, "volume_ratio"] = (data["Volume"][day:].values - data["Volume"][:-day].values) / data["Volume"][:-day].values
        return data

    @staticmethod
    def get_momentum(data):
        data.loc[1:, "momentum1"] = (data["Close"][1:].values / data["Close"][:-1].values) 
        data.loc[10:, "momentum10"] = (data["Close"][10:].values / data["Close"][:-10].values) 
        data.loc[30:, "momentum30"] = (data["Close"][30:].values / data["Close"][:-30].values) 
        data.loc[50:, "momentum50"] = (data["Close"][50:].values / data["Close"][:-50].values) 
        data.loc[70:, "momentum70"] = (data["Close"][70:].values / data["Close"][:-70].values) 
        return data

if __name__ == "__main__":
    path1 = "./Data/HA"
    path2 = "./Data/WBA"
    path3 = "./Data/COST"
    path4 = "./Data/INCY"
    path5 = "./Data/TIGO"
    path6 = "./Data/AAPL"
    path7 = "./Data/JD"

    # TIGO, REGN, INCY
    path_list = [path1, path2, path3, path4, path5, path6, path7]
    train_start = "2014-01-01"
    train_end = "2022-01-01"
    test_start = "2020-06-02"

    dm = DataManager(path_list, train_start, train_end, test_start)
    train_data, test_data = dm.get_data(path_list[6])
    train_data_tensor, test_data_tensor = dm.get_data_tensor()

    import matplotlib.pyplot as plt
    plt.plot(test_data["Price"].values)
    plt.show()
