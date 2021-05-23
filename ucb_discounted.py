from SMPyBandits.Policies.DiscountedUCB import DiscountedUCB
import numpy as np

from tqdm import trange
from matplotlib import pyplot as plt
import yaml

with open("variables.yaml") as f:
    d = yaml.safe_load(f)
    interval = d["interval"]
    max_val = d["max_val"]
    min_val = d["min_val"]

# arm_rewards = {
#     0: np.hstack((.7 * np.ones(interval), 0.3 * np.ones(interval), 0.5 * np.ones(interval), 0.3 * np.ones(interval))),
#     1: np.hstack((.3 * np.ones(interval), 0.3 * np.ones(interval), 0.6 * np.ones(interval), 0.7 * np.ones(interval))),
#     2: np.hstack((.5 * np.ones(interval), 0.7 * np.ones(interval), 0.3 * np.ones(interval), 0.4 * np.ones(interval))),
#     3: np.hstack((.1 * np.ones(interval), 0.1 * np.ones(interval), 1.0 * np.ones(interval), 0.1 * np.ones(interval))),
# }
arm_rewards = {
    0: np.hstack((max_val * np.ones(interval), min_val * np.ones(interval), max_val * np.ones(interval), min_val * np.ones(interval), max_val * np.ones(interval), min_val * np.ones(interval),)),
    1: np.hstack((min_val * np.ones(interval), max_val * np.ones(interval), min_val * np.ones(interval), max_val * np.ones(interval), min_val * np.ones(interval), max_val * np.ones(interval),)),
}

arm_colors = {
    0: "mediumseagreen",
    1: "tomato",
    2: "gold",
    3: "blue",
}

noise_std = 0.05

HORIZON = len(arm_rewards[0])
N_ARMS = len(arm_rewards)


algo = DiscountedUCB(N_ARMS)
algo.startGame()


arm_choices = np.zeros(HORIZON)
seen_rewards = np.zeros(HORIZON)
arm_means = np.zeros((N_ARMS, HORIZON))
ucbs = np.zeros((N_ARMS, HORIZON))
regret_sum = 0
regret = np.zeros(HORIZON)

for t in trange(HORIZON):
    noises = np.random.normal(0.0, scale=noise_std, size=N_ARMS)

    arm_idx = algo.choice()
    reward_at_idx = arm_rewards[arm_idx][t] + noises[arm_idx]

    algo.getReward(arm_idx, reward_at_idx)

    arm_choices[t] = arm_idx
    seen_rewards[t] = reward_at_idx
    algo.computeAllIndex()
    arm_means[:, t] = algo.discounted_rewards / algo.discounted_pulls
    ucbs[:, t] = algo.index

    regret_sum += max(*[arm_rewards[k][t] for k in arm_rewards.keys()]) - reward_at_idx
    regret[t] = regret_sum

_, ax = plt.subplots(2, 1, figsize=(10, 14), sharex=True)
for arm_idx in arm_rewards.keys():
    ax[0].plot(np.arange(HORIZON), arm_means[arm_idx], c=arm_colors[arm_idx], label=f"Arm: {arm_idx}", linewidth=4)
    ax[0].plot(np.arange(HORIZON), ucbs[arm_idx], c=arm_colors[arm_idx], label=f"UCB Arm: {arm_idx}", linewidth=4, linestyle="--")

ax[0].legend()
ax[0].set_ylabel("Empirical mean of arm & UCB")

for arm_idx, weight in arm_rewards.items():
    ax[1].plot(np.arange(HORIZON), weight, label=f"Arm: {arm_idx}", c=arm_colors[arm_idx], linewidth=4)
ax[1].legend()
ax[1].set_ylabel("True Rewards per arm")
ax[1].set_xlabel("Time")

plt.suptitle("Discounted UCB Algorithm Estimated Arm means")
plt.savefig("plots/ucb_discount_arms.png", bbox_inches="tight")
plt.savefig("plots/ucb_discount_arms.pdf", bbox_inches="tight")
plt.clf()
plt.cla()

# Plot regret
np.save("logs/ucb_discounted_regret.npy", regret)
fig = plt.figure(figsize=(10, 10))
plt.plot(np.arange(HORIZON), regret, label="Discounted UCB", linewidth=4)
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Discounted UCB Regret")
plt.legend()
plt.savefig("plots/ucb_discounted_regrets.pdf", bbox_inches="tight")
plt.savefig("plots/ucb_discounted_regrets.png", bbox_inches="tight")
