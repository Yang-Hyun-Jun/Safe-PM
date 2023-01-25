import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils

class Viz:
    def __init__(self):
        pass

    def show(self, seed_from, seed_to):
        SAVE_DIR = "/Users/mac/Desktop/Work Space/Safe PM 2/Metrics"
        mean_datas = []
        mode_datas = []
        BH_datas = []
        cppi_datas = []

        for i in range(seed_from, seed_to+1):
            path1 = SAVE_DIR + f"/seed{i}/Profitloss_Test_mean"
            path2 = SAVE_DIR + f"/seed{i}/Profitloss_Test_mode"
            path3 = SAVE_DIR + f"/seed{i}/Profitloss_Test_BH"
            path4 = SAVE_DIR + f"/seed{i}/Profitloss_Test_cppi"

            data1 = pd.read_csv(path1, index_col=0)["Profitloss"].to_numpy()
            data2 = pd.read_csv(path2, index_col=0)["Profitloss"].to_numpy()
            data3 = pd.read_csv(path3, index_col=0)["Profitloss"].to_numpy()
            data4 = pd.read_csv(path4, index_col=0)["Profitloss"].to_numpy()

            mean_datas.append(data1.reshape(-1,1))
            mode_datas.append(data2.reshape(-1,1))
            BH_datas.append(data3.reshape(-1,1))
            cppi_datas.append(data4.reshape(-1,1))

        expect_mean = np.mean(np.concatenate(mean_datas, axis=-1), axis=-1)
        expect_mode = np.mean(np.concatenate(mode_datas, axis=-1), axis=-1)
        expect_BH = np.mean(np.concatenate(BH_datas, axis=-1), axis=-1)
        expect_cppi = np.mean(np.concatenate(cppi_datas, axis=-1), axis=-1)

        std_mean = np.std(np.concatenate(mean_datas, axis=-1), axis=-1)
        std_mode = np.std(np.concatenate(mode_datas, axis=-1), axis=-1)
        std_BH = np.std(np.concatenate(BH_datas, axis=-1), axis=-1)
        std_cppi = np.std(np.concatenate(cppi_datas, axis=-1), axis=-1)

        beta = 0.1
        cl1 = "C3"
        cl2 = "C0"
        cl3 = "C2"
        cl4 = "C4"

        plt.figure(figsize=(5, 5))
        plt.fill_between(x=np.arange(expect_mean.shape[0]), y1=expect_mean + beta * std_mean, y2=expect_mean - beta * std_mean, alpha=0.3, color=cl1)
        plt.plot(expect_mean, label="mean", color=cl1)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)

        plt.fill_between(x=np.arange(expect_mode.shape[0]), y1=expect_mode + beta * std_mode, y2=expect_mode - beta * std_mode, alpha=0.3, color=cl2)
        plt.plot(expect_mode, label="mode", color=cl2)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)

        plt.fill_between(x=np.arange(expect_BH.shape[0]), y1=expect_BH + beta * std_BH, y2=expect_BH - beta * std_BH, alpha=1, color=cl3)
        plt.plot(expect_BH, label="BH", color=cl3)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)

        plt.fill_between(x=np.arange(expect_cppi.shape[0]), y1=expect_cppi + beta * std_cppi, y2=expect_cppi - beta * std_cppi, alpha=1, color=cl4)
        plt.plot(expect_cppi, label="CPPI", color=cl4)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)

        plt.grid(True, color="gray", alpha=0.5)
        plt.legend(fontsize=18)
        plt.ylabel("Profitloss", fontsize=20)
        plt.xlabel("Test timesteps", fontsize=20)
        plt.show()

        print("mean profitloss:", expect_mean[-1])
        print("mode profitloss:", expect_mode[-1])
        print("BH profitloss:", expect_BH[-1])
        print("CPPI profitloss:", expect_cppi[-1])
