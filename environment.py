"""
Observation의 Shape은 (S, F)
F: Num of features
S: Num of Stocks 
"""

class Environment:
    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = 0
        self.price_column = -1

    def reset(self):
        self.observation = None
        self.idx = 0

    def observe(self):
        if len(self.chart_data)-1 >= self.idx:
            self.observation = self.chart_data[self.idx]
            self.observation_train = self.observation[:self.price_column] 
            self.idx += 1
            return self.observation_train.transpose()
        else:
            return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.price_column]
        return None

