import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./data/backtest/2024-06-11-23-13-42.csv')
parser.add_argument('--sort', action="store_true")

in_args = parser.parse_args()
df = pd.read_csv(in_args.input)

sns.displot(data=df, x="pnl",bins=100)
# plt.xlim()

if in_args.sort:
    df.sort_values(by=['pnl'],inplace=True)
    df.dropna(inplace=True)
    df = df[df['pnl'] != 0]
    df.to_csv(in_args.input)
    
print(f"min {df['pnl'].min()}, max{df['pnl'].max()} mean{df['pnl'].mean()}")
plt.show()