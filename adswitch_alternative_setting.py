from adswitch_custom import AdSwitchCustom
import numpy as np
from typing import Set

from tqdm import trange
from matplotlib import pyplot as plt
from collections import defaultdict


interval = 200
max_arm_0_val = 3
min_arm_0_val = -1

max_arm_1_val = 2
min_arm_1_val = -3


# Setup
# arm_rewards = {
#     0: np.hstack((.7 * np.ones(interval), 0.3 * np.ones(interval), 0.5 * np.ones(interval), 0.3 * np.ones(interval))),
#     1: np.hstack((.3 * np.ones(interval), 0.3 * np.ones(interval), 0.6 * np.ones(interval), 0.7 * np.ones(interval))),
#     2: np.hstack((.5 * np.ones(interval), 0.7 * np.ones(interval), 0.3 * np.ones(interval), 0.4 * np.ones(interval))),
#     3: np.hstack((.1 * np.ones(interval), 0.1 * np.ones(interval), 1.0 * np.ones(interval), 0.1 * np.ones(interval))),
# }
arm_rewards = {
    0: np.hstack((max_arm_0_val * np.ones(interval), min_arm_0_val * np.ones(interval), min_arm_0_val * np.ones(interval), max_arm_0_val * np.ones(interval),)),
    1: np.hstack((min_arm_1_val * np.ones(interval), min_arm_1_val * np.ones(interval), max_arm_1_val * np.ones(interval), max_arm_1_val * np.ones(interval),)),
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
bad_arms: Set[int] = set()
bad_arm_add_hist = defaultdict(list)

arm_choices = np.zeros(HORIZON)
seen_rewards = np.zeros(HORIZON)
S_sets = {}

regret_sum = 0
regret = np.zeros(HORIZON)

for t in trange(HORIZON):
    noises = np.random.normal(0.0, scale=noise_std, size=N_ARMS)

    arm_idx = algo.choice()
    reward_at_idx = arm_rewards[arm_idx][t] + noises[arm_idx]

    algo.getReward(arm_idx, reward_at_idx)

    starts.add(algo.start_of_episode)

    if algo.set_S:
        S_sets[t] = algo.set_S.copy()

    bad_arms_t = set(range(N_ARMS)).difference(algo.set_GOOD)
    for arm in bad_arms_t.difference(bad_arms):
        bad_arm_add_hist[arm].append(t)
    bad_arms = bad_arms_t

    arm_choices[t] = arm_idx
    seen_rewards[t] = reward_at_idx

    regret_sum += max(*[arm_rewards[k][t] for k in arm_rewards.keys()]) - reward_at_idx
    regret[t] = regret_sum


_, ax = plt.subplots(2, 1, figsize=(10, 14), sharex=True)
# for arm_idx in arm_rewards.keys():
#     means_to_t = np.cumsum(seen_rewards * (arm_choices == arm_idx)) / np.cumsum(arm_choices == arm_idx)
#     ax[0].plot(np.arange(HORIZON), means_to_t, c=arm_colors[arm_idx], label=f"Arm: {arm_idx}", linewidth=4)

print(starts)
if 0 in starts:
    starts.remove(0)
if starts:
    starts = np.sort(list(starts))
    for it, start in enumerate(starts):
        if it == 0:
            ax[0].axvline(start, color="black", linestyle="dotted", label="Episode starts", linewidth=4)
        else:
            ax[0].axvline(start, color="black", linestyle="dotted", linewidth=4)

    starts = list(starts)
    starts.append(HORIZON)
    starts.insert(0, 0)
    for it, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        for arm_idx in arm_rewards.keys():
            means_to_t = np.cumsum(seen_rewards[start:end] * (arm_choices[start:end] == arm_idx)) / np.cumsum(arm_choices[start:end] == arm_idx)
            if it == 0:
                ax[0].plot(np.arange(start, end), means_to_t, c=arm_colors[arm_idx], label=f"Arm: {arm_idx}", linewidth=4)
            else:
                ax[0].plot(np.arange(start, end), means_to_t, c=arm_colors[arm_idx], linewidth=4)



if bad_arm_add_hist:
    for arm, ts in bad_arm_add_hist.items():
        for t in ts:
            ax[0].axvline(t, color=arm_colors[arm], linestyle="dotted", linewidth=4)

ax[0].legend()
ax[0].set_ylabel("Empirical mean of arm")

for arm_idx, weight in arm_rewards.items():
    ax[1].plot(np.arange(HORIZON), weight, label=f"Arm: {arm_idx}", c=arm_colors[arm_idx], linewidth=4)
ax[1].legend()
ax[1].set_ylabel("True Rewards per arm")
ax[1].set_xlabel("Time")

plt.suptitle("AdSwitch Algorithm Estimated Arm means")
plt.savefig("plots/adswitch_alt_arms.png", bbox_inches="tight")
plt.savefig("plots/adswitch_alt_arms.pdf", bbox_inches="tight")
plt.clf()
plt.cla()

# Plot regret
np.save("logs/adswitch_alt_regret.npy", regret)
fig = plt.figure(figsize=(10, 10))
plt.plot(np.arange(HORIZON), regret, label="AdSwitch", linewidth=4)
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("AdSwitch Regret")
plt.legend()
plt.savefig("plots/adswitch_alt_regrets.pdf", bbox_inches="tight")
plt.savefig("plots/adswitch_alt_regrets.png", bbox_inches="tight")
