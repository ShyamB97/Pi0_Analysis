#%%
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost.core import _prediction_output
from xgboost.training import train
import xgboost as xgb
import BDT_regressor
import Plots
import sys
import os
import time

# Disable terminal output
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore terminal output
def enablePrint():
    sys.stdout = sys.__stdout__


def PlotTrends():
    """
    Plot how BDT paramters vary with average prediction
    """
    for i in range(len(linspace)):
        print("parameter :" + str(i))
        param = param_default
        prediction = []
        
        for j in range(n):
            print("iteration: " + str(j))
            param[i][1] = linspace[i][j]
            p, _, _, _ = BDT_regressor.Run(test, training, n=1000, predict="invariant mass (GeV)", features=features, param=param)
            prediction.append(np.mean(p))
        
        plt.figure()
        plt.plot(linspace[i], prediction)
        plt.xlabel(i)
    plt.show()


# Booster parameter
param_default = [
    ['eta', 0.3],
    ['gamma', 0],
    ['max_depth', 6],
    ['subsample', 1],
    ['colsample_bytree', 1],
    ['objective', 'reg:gamma'],
    ['eval_metric', 'error'],
    ['eval_metric', 'logloss'],
    ['eval_metric', 'rmse']
]

training = pd.read_csv("BDT_input/features_new/diphoton_truth_hits.txt")
test = pd.read_csv("BDT_input/features_new/pi0_0p5GeV_hits.txt")

features = training.columns[1:]
features = [feature for feature in features if "true" not in feature]
features = [feature for feature in features if "beam" not in feature] # no beam yet
features.pop( features.index("invariant mass (GeV)") )

n_0 = 10
n_1 = 10
n_2 = 7
n_3 = 5
n_4 = 10
eta = np.linspace(0.05, 1, n_0)
gamma = np.linspace(0, 2, n_1)
max_depth = np.linspace(3, 10, n_2, dtype=int)
subsample = np.linspace(0.5, 1, n_3)
colsample_bytree = np.linspace(0.2, 1, n_4)

linspace = [eta, gamma, max_depth, subsample, colsample_bytree]

iter_total = n_0 * n_1 * n_2 * n_3 * n_4
iter_times = []
configurations = []
_iter = 0
for i in range(n_0):
    for j in range(n_1):
        for k in range(n_2):
            for l in range(n_3):
                for m in range(n_4):
                    iter_start = time.time()
                    _iter += 1
                    print("configuration: " + str(_iter))
                    
                    param = param_default
                    param[0][1] = linspace[0][i]
                    param[1][1] = linspace[1][j]
                    param[2][1] = linspace[2][k]
                    param[3][1] = linspace[3][l]
                    param[4][1] = linspace[4][m]
                    p, _, _, _ = BDT_regressor.Run(test, training, n=1000, predict="invariant mass (GeV)", features=features, param=param, _print=False)
                    configurations.append( [linspace[0][i], linspace[1][j], linspace[2][k], linspace[3][l], linspace[4][m], np.mean(p), np.std(p)] )
                    os.system( 'clear' )                    
                    iter_time = time.time() - iter_start
                    print("computation time: " + str(iter_time) + "s")
                    iter_times.append(iter_time)
                    iter_average_time = np.mean(iter_times)
                    iter_remaining_time = iter_average_time * (iter_total - _iter)
                    print("time remaining: " + str( iter_remaining_time ) + "s")
                    print("total configurations: " + str(iter_total))
                    if _iter == iter_total:
                        print("completed!")



columns = ["eta", "gamma", "max_depth", "subsample", "colsample_bytree", "average", "deviation"]
df = pd.DataFrame(configurations, columns=columns)
print("saving to file")
df.to_csv("configurations.csv")
# %%
