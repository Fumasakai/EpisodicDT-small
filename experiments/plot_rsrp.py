import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/raw/rsrp_sample.csv")

plt.plot(df["time"], df["rsrp"])

plt.xlabel("Time")
plt.ylabel("RSRP [dBm]")
plt.title("Generated RSRP")

plt.grid()
plt.show()
