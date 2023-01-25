import torch
import numpy as np
import pandas as pd

from collections import deque
from replaymemory import ReplayMemory
from environment import Environment
from agent import Agent
from network import Network
from metrics import Metrics

class Trainer:
    """
    s: feature state
    p: portfolio state
    a: action
    r: reward
    c: cost 
    """

    def __init__(self,
                 lr1, lr2, term, freq,
                 tau, delta, alpha, gamma, fee,
                 batch_size, memory_size, cons,
                 data, balance, episode, K, F,  
                 min_trading_price, max_trading_price):

        assert min_trading_price >= 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price

        self.net = Network(K=K, F=F)
        self.target_net = Network(K=K, F=F)
        self.environment = Environment(data)
        self.memory = ReplayMemory(memory_size)
        self.metrics = Metrics()

        self.K = K
        self.F = F
        self.freq = freq
        self.cons = cons
        self.balance = balance
        self.episode = episode
        self.batch_size = batch_size
        self.data = data

        self.agent = Agent(lr1=lr1, lr2=lr2, term=term,
                           environment=self.environment,
                           cost=fee, alpha=alpha, tau=tau,
                           delta=delta, K=K, gamma=gamma,
                           net = self.net, target_net=self.target_net, 
                           min_trading_price=min_trading_price,
                           max_trading_price=max_trading_price)

    def reset(self):
        self.agent.set_balance(self.balance)
        self.environment.reset()
        self.agent.reset()
        self.metrics.reset()

    def train(self):
        cumrs_window = deque(maxlen=100)
        cumcs_window = deque(maxlen=100)

        for epi in range(self.episode):
            self.reset()
            cum_r = 0
            cum_c = 0 
            steps_done = 0
            costs = []

            prices = self.agent.environment.observe()
            portfolio = self.agent.portfolio
            cushion = self.agent.get_cushion()
            state = self.make_state(prices, portfolio, cushion)

            while True:
                action, sample, log_prob = self.agent.get_action(torch.tensor(state).float().view(1,self.K+1,-1))
                n_prices, n_portfolio, reward_, cost, done = self.agent.step(action, abs(action))
                n_cushion = self.agent.get_cushion()
                next_state = self.make_state(n_prices, n_portfolio, n_cushion)
                reward = reward_ - self.agent.lam * cost if self.cons else reward_ 

                experience = (torch.tensor(state).float().view(1,self.K+1,-1),
                              torch.tensor(sample).float().view(1,-1),
                              torch.tensor(reward).float().view(1,-1),
                              torch.tensor(next_state).float().view(1,self.K+1,-1),
                              torch.tensor(log_prob).float().view(1,-1),
                              torch.tensor(done).float().view(1,-1))

                self.memory.push(experience)
                cum_r += reward_ 
                cum_c += cost
                steps_done += 1
                state = next_state
                costs.append(cost)

                if len(self.memory) >= self.batch_size:
                    sampled_exps = self.memory.sample(self.batch_size)
                    sampled_exps = self.prepare_training_inputs(sampled_exps)
                    self.agent.update(*sampled_exps)
                    self.agent.soft_target_update(self.agent.net.parameters(), self.agent.target_net.parameters())
                    
                if steps_done % self.freq == 0:
                    const, lam_grad = self.agent.update_lam(costs)

                if epi == range(self.episode)[-1]:
                    self.metrics.portfolio_values.append(self.agent.portfolio_value)
                    self.metrics.profitlosses.append(self.agent.profitloss)
                    self.metrics.cum_fees.append(self.agent.cum_fee)
                    self.metrics.cash.append(self.agent.portfolio[0])

                if done:
                    const, lam_grad = self.agent.update_lam(costs)
                    value = self.agent.net.value(experience[0]).detach()
                    alpha = self.agent.net.alpha(experience[0]).detach()
                    cumrs_window.append(cum_r)
                    cumcs_window.append(cum_c)
                    score_r = np.mean(cumrs_window)
                    score_c = np.mean(cumcs_window)

                    print(f"epi:{epi}")
                    print(f"cum cost:{cum_c}")
                    print(f"cum reward:{cum_r}")
                    print(f"score r:{score_r}")
                    print(f"score c:{score_c}")
                    print(f"cushion:{n_cushion}")
                    print(f"a:{action}")
                    print(f"c:{cost}")
                    print(f"alpha:{alpha}")
                    print(f"log prob:{log_prob}")
                    print(f"value:{value}")
                    print(f"const:{const}")
                    print(f"lam:{self.agent.lam}")
                    print(f"lam_grad:{lam_grad}")
                    print(f"cum_fee:{self.agent.cum_fee}")
                    print(f"portfolio:{self.agent.portfolio}")
                    print(f"profitloss:{self.agent.profitloss}")
                    print(f"loss:{self.agent.loss}\n")

                    if epi == range(self.episode)[-1]:
                        self.metrics.get_profitlosses()
                        self.metrics.get_portfolio_values()
                        self.metrics.get_fees()
                        self.metrics.get_cash()

                    break
     
    def save_model(self, net_path):
        torch.save(self.agent.net.state_dict(), net_path)

    def make_state(self, prices, portfolio, cushion):
        prices = prices.reshape(self.K, self.F)
        portfolio = portfolio.reshape(self.K+1, 1)
        cushions = np.ones((self.K+1, 1)) * cushion
        cash_price = np.ones(shape=(1, self.F)) * 0.1
        prices = np.concatenate([cash_price, prices], axis=0)
        state = np.concatenate([prices, portfolio, cushions], axis=1)
        return state 
    
    @staticmethod
    def prepare_training_inputs(sampled_exps):
        """
        s, a, r, c, ns, log_probs, dones
        """
        num = 6
        x = [[] for _ in range(num)]

        for sampled_exp in sampled_exps:
            for i in range(num):
                x[i].append(sampled_exp[i])
        for i in range(num):
            x[i] = torch.cat(x[i], dim=0).float()

        return x
    
    @staticmethod
    def get_mdd(array):
        df = pd.DataFrame({"PV":array})
        df["전고점"] = df["PV"].cummax()
        df["DrawDown"] = (1-df["PV"] / df["전고점"]) * 100
        MDD = df["DrawDown"].max()
        return MDD

