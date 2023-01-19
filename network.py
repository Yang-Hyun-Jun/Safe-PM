import torch
import torch.nn as nn
import numpy as np

from torch.distributions.dirichlet import Dirichlet

class Network(nn.Module):
    def __init__(self, K, F):
        super().__init__()

        self.K = K
        self.F = F

        self.score_net = nn.Sequential(
            nn.Linear((self.F + 2) * (self.K + 1), 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.K+1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(K+1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

        self.const_net = nn.Sequential(
            nn.Linear(K+1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        """
        state: [Batch Num, K, 3]
        3: 종목 당 가격 변화율, 종목 당 포트폴리오 비율, 현재 쿠션
        """
        return self.score_net(x)
    
    def const(self, state):
        """
        Expected Sum of Cost
        """
        state = state.reshape(-1, (self.F + 2) * (self.K + 1))
        scores = self(state).reshape(-1, self.K+1)
        c = self.const_net(scores)
        return c

    def value(self, state):
        """
        Critic의 Value
        """
        state = state.reshape(-1, (self.F + 2) * (self.K + 1))
        scores = self(state).reshape(-1, self.K+1)
        v = self.value_net(scores)
        return v

    def alpha(self, state):
        """
        Dirichlet Dist의 Concentration Parameter
        """
        state = state.reshape(-1, (self.F + 2) * (self.K + 1))
        scores = self(state).reshape(-1, self.K+1)
        alpha = torch.exp(scores) + 1
        return alpha

    def log_prob(self, state, portfolio):
        """
        Dirichlet Dist에서 샘플의 log_prob
        """
        alpha = self.alpha(state)
        dirichlet = Dirichlet(alpha[0])
        return dirichlet.log_prob(portfolio)

    def sampling(self, state, repre=False):
        """
        Dirichlet Dist에서 포트폴리오 샘플링
        """
        alpha = self.alpha(state).detach()
        dirichlet = Dirichlet(alpha[0])
        sampled_p = None 
        
        B = alpha.shape[0]  # Batch num
        N = alpha.shape[1]  # Asset num + 1

        if repre == "mean":
            sampled_p = dirichlet.mean

        elif repre == "mode":
            sampled_p = (alpha[0]-1)/(torch.sum(alpha[0]) - N)

        elif repre == "BH":
            sampled_p = torch.ones(size=(N,)) / N

        elif repre == "cppi":
            param = 5.0
            cash = np.exp(-param*max(state[:,0,-1], 0))
            sampled_p = torch.ones(size=(N,)) * (1-cash) / (N-1)
            sampled_p[0] = cash

        elif not repre:
            sampled_p = dirichlet.sample([1])[0]
        
        return sampled_p
        

if __name__ == "__main__":

    import numpy as np
    from torch.distributions import Normal
    
    K = 3
    F = 1

    cushion = 0.2
    prices = np.random.random(size=(K, F))
    portfolio = np.random.random(size=(K+1, 1))

    cushions = np.ones(shape=(K+1, 1)) * (0.5)
    cash_price = np.ones(shape=(1, F))
    prices = np.concatenate([cash_price, prices], axis=0)
    state = np.concatenate([prices, portfolio, cushions], axis=1)    
    state = torch.FloatTensor(state).view(1, K+1, -1)

    network = Network(K=K, F=F)
    value = network.value(state)
    alpha = network.alpha(state)
    mean = network.sampling(state, "mean")
    mode = network.sampling(state, "mode")
    rule = network.sampling(state, "cppi")
    BH = network.sampling(state, "BH")
    log_pi = network.log_prob(state, BH)
    
    print("state", state)
    print("alpha", alpha)
    print("value", value)
    print("log_pi", log_pi)
    print("mean", mean)
    print("mode", mode)
    print("BH", BH)
    print(np.exp(-2.3*max(0.8, 0)))
    print("cppi", rule)