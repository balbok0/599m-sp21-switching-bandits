import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 10))
for file_name, label in [
    ("adswitch", "AdSwitch"),
    ("exp3", "EXP3"),
    ("exp3s", "EXP3.S"),
    ("ucb_discounted", "Discounted UCB"),
    ("ucb_sliding_window", "UCB SW"),
]:
    regret = np.load(f"logs/{file_name}_regret.npy")
    plt.plot(np.arange(len(regret)), regret, label=label, linewidth=4)
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret Comparison")
plt.legend()
plt.savefig("plots/regrets.pdf", bbox_inches="tight")
plt.savefig("plots/regrets.png", bbox_inches="tight")
