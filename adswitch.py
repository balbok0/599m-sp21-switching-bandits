from adswitch_custom import AdSwitchCustom, AdSwitchNew
import numpy as np

from tqdm import trange
from matplotlib import pyplot as plt

# Setup
interval = 100
# arm_rewards = {
#     0: np.hstack((.7 * np.ones(interval), 0.3 * np.ones(interval), 0.5 * np.ones(interval), 0.3 * np.ones(interval))),
#     1: np.hstack((.3 * np.ones(interval), 0.3 * np.ones(interval), 0.6 * np.ones(interval), 0.7 * np.ones(interval))),
#     2: np.hstack((.5 * np.ones(interval), 0.7 * np.ones(interval), 0.3 * np.ones(interval), 0.4 * np.ones(interval))),
#     3: np.hstack((.1 * np.ones(interval), 0.1 * np.ones(interval), 1.0 * np.ones(interval), 0.1 * np.ones(interval))),
# }
arm_rewards = {
    0: np.hstack((.9 * np.ones(2 * interval), .1 * np.ones(2 * interval))),
    1: np.hstack((.1 * np.ones(2 * interval), .9 * np.ones(2 * interval))),
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

delta_t = 1
delta_s = 1

algo = AdSwitchCustom(N_ARMS, horizon=HORIZON, delta_t=delta_t, delta_s=delta_s)  # , delta_s=4, delta_t=10
# algo = AdSwitchNew(N_ARMS, horizon=HORIZON, delta_t=delta_t, delta_s=delta_s)  # , delta_s=4, delta_t=10
algo.startGame()

starts = set()

arm_choices = np.zeros(HORIZON)
seen_rewards = np.zeros(HORIZON)
S_sets = {}

for t in trange(HORIZON):
    noises = np.random.normal(0.0, scale=noise_std, size=N_ARMS)

    arm_idx = algo.choice()
    reward_at_idx = arm_rewards[arm_idx][t] + noises[arm_idx]

    algo.getReward(arm_idx, reward_at_idx)

    starts.add(algo.start_of_episode)

    if algo.set_S:
        S_sets[t] = algo.set_S.copy()

    arm_choices[t] = arm_idx
    seen_rewards[t] = reward_at_idx


_, ax = plt.subplots(2, 1, figsize=(10, 14), sharex=True)
for arm_idx in arm_rewards.keys():
    means_to_t = np.cumsum(seen_rewards * (arm_choices == arm_idx)) / np.cumsum(arm_choices == arm_idx)
    ax[0].plot(np.arange(HORIZON), means_to_t, c=arm_colors[arm_idx], label=f"Arm: {arm_idx}", linewidth=4)

starts.remove(0)
if starts:
    ax[0].axvline(list(starts), color="black", linestyle="dotted", label="Episode starts", linewidth=4)

ax[0].legend()
ax[0].set_ylabel("Empirical mean of arm")

for arm_idx, weight in arm_rewards.items():
    ax[1].plot(np.arange(HORIZON), weight, label=f"Arm: {arm_idx}", c=arm_colors[arm_idx], linewidth=4)
ax[1].legend()
ax[1].set_ylabel("True Rewards per arm")
ax[1].set_xlabel("Time")

plt.suptitle("AdSwitch Algorithm Estimated Arm means")
plt.savefig("plots/adswitch_arms.png", bbox_inches="tight")
plt.savefig("plots/adswitch_arms.pdf", bbox_inches="tight")
