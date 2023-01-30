import torch
import numpy as np
from trainer import Trainer

class Tester(Trainer):
    def __init__(self, **args):
        Trainer.__init__(self, **args)

    def test(self, path=None, tests=["BH", "mean", "mode", "cppi"]):
        for test in tests:
            self.agent.net.load_state_dict(torch.load(path))
            self.reset()

            cum_r = 0
            cum_c = 0 
            steps_done = 0            

            prices = self.agent.environment.observe()
            portfolio = self.agent.portfolio
            cushion = self.agent.get_cushion()
            state = self.make_state(prices, portfolio, cushion)       

            while True:
                mode = "_Test_" + test
                action, sample, log_prob = self.agent.get_action(torch.tensor(state).float().view(1,self.K+1,-1), test)
                n_prices, n_portfolio, reward, cost, done = self.agent.step(action, abs(action))
                n_cushion = self.agent.get_cushion()
                next_state = self.make_state(n_prices, n_portfolio, n_cushion)
                # print(self.agent.portfolio, action)
                
                cum_r += reward
                cum_c += cost
                steps_done += 1
                state = next_state

                self.metrics.portfolio_values.append(self.agent.portfolio_value)
                self.metrics.profitlosses.append(self.agent.profitloss)
                self.metrics.cum_fees.append(self.agent.cum_fee)
                self.metrics.cash.append(self.agent.portfolio[0])

                if done:
                    print(f"epi:" + mode)
                    print(f"cum cost:{cum_c}")
                    print(f"cum reward:{cum_r}")
                    print(f"cushion:{n_cushion}")
                    print(f"a:{action}")
                    print(f"c:{cost}")
                    print(f"cum_fee:{self.agent.cum_fee}")
                    print(f"portfolio:{self.agent.portfolio}")
                    print(f"profitloss:{self.agent.profitloss}\n")
                    self.metrics.get_profitlosses(mode=mode)
                    self.metrics.get_portfolio_values(mode=mode)
                    self.metrics.get_fees(mode=mode)
                    self.metrics.get_cash(mode=mode)
                    break
