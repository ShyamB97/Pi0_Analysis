import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Plots

file = "configurations.csv"
subDirectory = "BDT_config_analysis/raw/"
os.makedirs(subDirectory, exist_ok=True)

df = pd.read_csv(file)
df = df[df["average"] < 1]
df = df[df["deviation"] < 1]

df = df[df["average"] < 0.14]
df = df[df["average"] > 0.13]



for column in df.columns:
    plt.figure()
    Plots.PlotHist(df[column], bins=20, xlabel=column)
    Plots.Save(column, subDirectory=subDirectory)