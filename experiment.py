from datamanager import DataManager
from trainer import DDT
import argparse
import numpy as np
import torch
import utils

# ["COST", "INCY"]
# ["TIGO", "BIDU"]
# ["COST", "INCY", "REGN"]
parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", default=["COST", "INCY"])
parser.add_argument("--train_start", type=str, default="2014-01-02")
parser.add_argument("--train_end", type=str, default="2019-12-31")
parser.add_argument("--test_start", type=str, default="2020-06-02")
parser.add_argument("--test_end", type=str, default="2021-12-31")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lr1", type=float, default=1e-4)
parser.add_argument("--lr2", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--fee", type=float, default=0.0000)
parser.add_argument("--delta", type=float, default=0.000)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--episode", type=float, default=500)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--batch_size", type=float, default=128)
parser.add_argument("--memory_size", type=float, default=10000)
parser.add_argument("--balance", type=float, default=12000)
parser.add_argument("--max_trading_price", type=float, default=400)
parser.add_argument("--min_trading_price", type=float, default=0)
parser.add_argument("--cons", type=bool, default=True)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
utils.SAVE_DIR += f"/seed{args.seed}"

paths = ["./Data/" + s for s in args.tickers]
datamanager = DataManager(paths, args.train_start, args.train_end, args.test_start, args.test_end)
train_data_tensor, test_data_tensor = datamanager.get_data_tensor()

K = train_data_tensor.shape[2]
F = train_data_tensor.shape[1]-1

parameters_train = {"lr1":args.lr1, 
                    "lr2":args.lr2, 
                    "tau":args.tau, 
                    "delta":args.delta, 
                    "alpha":args.alpha,
                    "gamma":args.gamma,
                    "K":K, "F":F, 
                    "fee":args.fee, 
                    "cons":args.cons,
                    "balance":args.balance, 
                    "episode":args.episode,
                    "train_data":train_data_tensor,
                    "test_data":test_data_tensor,
                    "min_trading_price":args.min_trading_price,
                    "max_trading_price":args.max_trading_price,
                    "batch_size":args.batch_size,
                    "memory_size":args.memory_size}

trainer = DDT(**parameters_train)
trainer.train()
trainer.test()
trainer.save_model(utils.SAVE_DIR + "/net.pth")

