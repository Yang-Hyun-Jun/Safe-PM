import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from collections import deque

class Agent(nn.Module):
    """
    거래 비용
    TRADING_CHARGE = 0.00015
    TRADING_TEX = 0.0025
    TRADING_CHARGE = 0.0
    TRADING_TEX = 0.0
    """

    def __init__(self, gamma:float, environment,
                 net:nn.Module, target_net:nn.Module,
                 K:int, cost:float, alpha:float, 
                 lr1:float, lr2:float, term:float,
                 tau:float, delta:float,  
                 min_trading_price:int,
                 max_trading_price:int):

        super().__init__()
        self.environment = environment
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        
        self.net = net
        self.target_net = target_net
        self.lr1 = lr1
        self.lr2 = lr2
        self.tau = tau
        self.lam = 0
        self.K = K
        self.term = term
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.loss = None

        self.net.load_state_dict(self.target_net.state_dict())
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr1)
        self.huber = nn.SmoothL1Loss()

        self.num_stocks = np.array([0] * self.K, dtype=int)
        self.portfolio = np.array([0] * (self.K+1), dtype=float)
        self.PVS = deque(maxlen=self.term)

        self.TRADING_CHARGE = cost
        self.TRADING_TEX = 0.0
        self.portfolio_value = 0
        self.initial_balance = 0
        self.balance = 0
        self.profitloss = 0
        self.cum_fee = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def reset(self):
        self.PVS = deque(maxlen=self.term)
        self.PVS.append(self.initial_balance)
        self.num_stocks = np.array([0] * self.K, dtype=int)
        self.portfolio = np.array([0] * (self.K+1), dtype=float)
        self.portfolio[0] = 1.0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.profitloss = 0
        self.cum_fee = 0

    def decide_trading_unit(self, confidence, price):
        trading_amount = self.portfolio_value * confidence
        trading_unit = int(np.array(trading_amount)/price)
        return trading_unit

    def validate_action(self, action, delta):
        m_action = action.copy()
        for i in range(action.shape[0]):
            if delta < action[i] <= 1:
                # 매수인 경우 적어도 1주를 살 수 있는지 확인
                if self.balance < self.environment.get_price()[i] * (1 + self.TRADING_CHARGE):
                    m_action[i] = 0.0 #Hold

            elif -1 <= action[i] < -delta:
                # 매도인 경우 주식 잔고가 있는지 확인
                if self.num_stocks[i] == 0:
                    m_action[i] = 0.0 #Hold
        return m_action

    def pi_operator(self, change_rate):
        pi_vector = np.zeros(len(change_rate) + 1)
        pi_vector[0] = 1
        pi_vector[1:] = change_rate + 1
        return pi_vector

    def get_action(self, s, repre=False):
        with torch.no_grad():
            self.net.eval()
            sample = self.net.sampling(s, repre)
            log_prob = self.net.log_prob(s, sample)
            sample = sample.numpy().reshape(self.portfolio.shape)
            log_prob = log_prob.numpy()
            action = (sample - self.portfolio)[1:]
            self.net.train()
        return action, sample, log_prob

    def get_portfolio_value(self, close_p1, close_p2, portfolio):
        close_p2 = np.array(close_p2)
        close_p1 = np.array(close_p1)
        change_rate = (close_p2 - close_p1)/close_p1
        pi_vector = self.pi_operator(change_rate)
        portfolio_value = np.dot(self.portfolio_value * portfolio, pi_vector)
        return portfolio_value

    def get_portfolio(self, close_p1, close_p2):
        close_p2 = np.array(close_p2)
        close_p1 = np.array(close_p1)
        change_rate = (close_p2 - close_p1)/close_p1
        pi_vector = self.pi_operator(change_rate)
        portfolio = (self.portfolio * pi_vector)/(np.dot(self.portfolio, pi_vector))
        return portfolio

    def get_cushion(self):
        param = 0.95
        param = 0.98
        base = param * self.initial_balance
        base = min(self.PVS)
        return self.portfolio_value / base - 1

    def get_reward(self, pv):
        reward = np.log(pv) - np.log(self.initial_balance)
        return reward

    def get_cost(self, mode):
        if mode == "pv":
            cost = 10 if self.portfolio_value <= min(self.PVS) else 0
        elif mode == "mdd":
            cost = 1 if self.get_mdd(self.PVS) >= 5 else 0
        elif mode == "profitloss":
            cost = 1 if self.profitloss <= -2.0 else 0 
        return cost

    def step(self, action, confidence):
        assert action.shape[0] == confidence.shape[0]
        assert 0 <= self.delta < 1

        fee = 0
        close_p1 = self.environment.get_price()
        m_action = self.validate_action(action, self.delta)
        self.portfolio_value_static_ = self.portfolio * self.portfolio_value

        # 종목별 매도 수행을 먼저한다.
        for i in range(m_action.shape[0]):
            p1_price = close_p1[i]

            if abs(m_action[i]) > 1.0:
                raise Exception(f"Action is out of bound: {m_action}")

            # Sell
            if -1 <= m_action[i] < -self.delta:
                cost = self.TRADING_TEX + self.TRADING_CHARGE
                trading_unit = self.decide_trading_unit(confidence[i], p1_price)
                trading_unit = min(trading_unit, self.num_stocks[i])
                invest_amount = p1_price * trading_unit

                fee += invest_amount * cost
                self.cum_fee += fee
                self.num_stocks[i] -= trading_unit
                self.balance += invest_amount * (1-cost)
                self.portfolio[0] += invest_amount * (1-cost)/self.portfolio_value
                self.portfolio[i+1] -= invest_amount/self.portfolio_value
                m_action[i] = -invest_amount/self.portfolio_value


        # 다음으로 종목별 매수 수행
        for i in range(m_action.shape[0]):
            p1_price = close_p1[i]

            if abs(m_action[i]) > 1.0:
                raise Exception(f"Action is out of bound: {m_action}")
                
            # Buy
            if self.delta < m_action[i] <= 1:
                cost = self.TRADING_CHARGE
                trading_unit = self.decide_trading_unit(confidence[i], p1_price)
                cal_balance = (self.balance - p1_price * trading_unit * (1+cost))

                # 돈 부족 한 경우
                if cal_balance < 0:
                    trading_unit = min(
                        int(self.balance / (p1_price * (1+cost))),
                        int(self.max_trading_price / (p1_price * (1+cost))))

                invest_amount = p1_price * trading_unit
                fee += invest_amount * cost
                self.cum_fee += fee
                self.num_stocks[i] += trading_unit
                self.balance -= invest_amount * (1+cost)
                self.portfolio[0] -= invest_amount * (1+cost)/self.portfolio_value
                self.portfolio[i+1] += invest_amount/self.portfolio_value
                m_action[i] = invest_amount/self.portfolio_value

            # Hold
            elif -self.delta <= m_action[i] <= self.delta:
                m_action[i] = 0.0

        # 거래로 인한 PV와 PF 변동 계산
        self.portfolio_value -= fee
        self.portfolio = self.portfolio / np.sum(self.portfolio) #sum = 1

        # 다음 Time step 으로 진행 함에 따라 생기는 가격 변동에 의한 PV와 PF 계산
        next_prices = self.environment.observe()
        close_p2 = self.environment.get_price()
        
        self.change = (np.array(close_p2)-np.array(close_p1))/np.array(close_p1)
        self.portfolio = self.get_portfolio(close_p1=close_p1, close_p2=close_p2)
        self.portfolio_value = self.get_portfolio_value(close_p1=close_p1, close_p2=close_p2, portfolio=self.portfolio)
        self.portfolio_value_static = np.dot(self.portfolio_value_static_, self.pi_operator(self.change))
        self.profitloss = ((self.portfolio_value / self.initial_balance) - 1)*100
        next_portfolio = self.portfolio

        self.PVS.append(self.portfolio_value)
        cost = 1 if self.get_mdd(self.PVS) >= 3 else 0
        reward = 0.0 * sum(self.portfolio[1:]) + self.get_reward(self.portfolio_value) 
        done = 1 if len(self.environment.chart_data)-1 <= self.environment.idx else 0 
        return next_prices, next_portfolio, reward, cost, done
    
    def get_mdd(self, array):
        df = pd.DataFrame({"PV":array})
        df["PreMax"] = df["PV"].cummax()
        df["DrawDown"] = (1-df["PV"] / df["PreMax"]) * 100
        MDD = df["DrawDown"].max()
        return MDD

    def update(self, s, p, r, ns, log_prob, done):
        eps_clip = 0.1
        log_prob = log_prob.view(-1, 1)
        log_prob_ = self.net.log_prob(s, p).view(-1, 1)
        ratio = torch.exp(log_prob_ - log_prob)

        # Critic loss
        with torch.no_grad():
            next_value = self.target_net.value(ns)
            v_target = r + self.gamma * next_value * (1-done)
            v_target = v_target * torch.clamp(ratio.detach(), 1-eps_clip, 1+eps_clip)

        value = self.net.value(s)
        v_loss = self.huber(value, v_target)

        # Actor loss
        td_advantage = r + self.gamma * self.net.value(ns) * (1-done) - value
        td_advantage = td_advantage.detach()
        surr1 = ratio * td_advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * td_advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        # Update
        self.loss = v_loss + actor_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_lam(self, costs):
        gam = self.gamma
        const = np.mean([c * (1-(gam)**i)/(1-gam) for i,c in enumerate(costs)])
        lam_grad = -(const - self.alpha)
        self.lam -= self.lr2 * lam_grad
        self.lam = max(self.lam, 0)
        return const, lam_grad

    def soft_target_update(self, params, target_params):
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def hard_target_update(self):
        self.net.load_state_dict(self.target_net.state_dict())


