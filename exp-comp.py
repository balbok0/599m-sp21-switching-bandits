from SMPyBandits.Policies.Exp3S import Exp3S
from SMPyBandits.Policies.Exp3 import Exp3
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

interval = 100
arm_rewards = {
    0: np.hstack((.7 * np.ones(interval), 0.3 * np.ones(interval), 0.5 * np.ones(interval), 0.3 * np.ones(interval))),
    1: np.hstack((.3 * np.ones(interval), 0.3 * np.ones(interval), 0.6 * np.ones(interval), 0.7 * np.ones(interval))),
    2: np.hstack((.5 * np.ones(interval), 0.7 * np.ones(interval), 0.3 * np.ones(interval), 0.4 * np.ones(interval))),
    3: np.hstack((.1 * np.ones(interval), 0.1 * np.ones(interval), 1.0 * np.ones(interval), 0.1 * np.ones(interval))),
}

# arm_rewards = {
#     0: np.hstack((.7 * np.ones(interval), 0.1 * np.ones(interval), 0.7 * np.ones(interval), 0.1 * np.ones(interval))),
#     1: np.hstack((.1 * np.ones(interval), 0.7 * np.ones(interval), 0.1 * np.ones(interval), 0.7 * np.ones(interval))),
# }
arm_colors = {
    0: "mediumseagreen",
    1: "tomato",
    2: "gold",
    3: "blue",
}

noise_std = 0.05

HORIZON = len(arm_rewards[0])
N_ARMS = len(arm_rewards)
GAMMA = 0.1
ALPHA_PLOT = 0.1


exp3s = Exp3S(nbArms=N_ARMS, horizon=HORIZON, gamma=GAMMA)
exp3s.startGame()

exp3 = Exp3(nbArms=N_ARMS, gamma=GAMMA)
exp3.startGame()


exp3s_weights = np.zeros((N_ARMS, HORIZON))
exp3s_regret = np.zeros(HORIZON)
exp3_weights = np.zeros((N_ARMS, HORIZON))
exp3_regret = np.zeros(HORIZON)


exp3s_regret_sum = 0
exp3_regret_sum = 0

for t in trange(HORIZON):
    noises = np.random.normal(0.0, scale=noise_std, size=N_ARMS)

    arm_idx = exp3s.choice()
    reward_at_idx = arm_rewards[arm_idx][t] + noises[arm_idx]

    exp3s.getReward(arm_idx, reward_at_idx)
    exp3s_weights[:, t] = exp3s.trusts

    exp3s_regret_sum += max(*[arm_rewards[k][t] for k in arm_rewards.keys()]) - reward_at_idx
    exp3s_regret[t] = exp3s_regret_sum

    arm_idx = exp3.choice()
    reward_at_idx = arm_rewards[arm_idx][t] + noises[arm_idx]

    exp3.getReward(arm_idx, reward_at_idx)
    exp3_weights[:, t] = exp3.trusts

    exp3_regret_sum += max(*[arm_rewards[k][t] for k in arm_rewards.keys()]) - arm_rewards[arm_idx][t]
    exp3_regret[t] = exp3_regret_sum


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15, 20))

times = np.arange(HORIZON)
for arm_idx, weight in enumerate(exp3s_weights):
    ax[0].plot(times, weight, label=f"Arm: {arm_idx}", c=arm_colors[arm_idx], linewidth=4)

xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()

for arm_idx, weight in enumerate(exp3_weights):
    ax[0].plot(times, weight, c=arm_colors[arm_idx], linewidth=4, alpha=ALPHA_PLOT)

ax[0].set_xlim(xlim)
ax[0].set_ylim(ylim)
ax[0].legend()
ax[0].set_ylabel("EXP3S Probabilites per arm")

for arm_idx, weight in arm_rewards.items():
    ax[1].plot(times, weight, label=f"Arm: {arm_idx}", c=arm_colors[arm_idx], linewidth=4)
ax[1].legend()
ax[1].set_ylabel("True Rewards per arm")

for arm_idx, weight in enumerate(exp3_weights):
    ax[2].plot(times, weight, label=f"Arm: {arm_idx}", c=arm_colors[arm_idx], linewidth=4)

xlim = ax[2].get_xlim()
ylim = ax[2].get_ylim()

for arm_idx, weight in enumerate(exp3s_weights):
    ax[2].plot(times, weight, c=arm_colors[arm_idx], linewidth=4, alpha=ALPHA_PLOT)

ax[2].set_xlim(xlim)
ax[2].set_ylim(ylim)

ax[2].legend()
ax[2].set_ylabel("EXP3 Probabilites per arm")
ax[2].set_xlabel("Time")

plt.suptitle("Arm probabilites EXP3 vs EXP3S")
plt.savefig("plots/exp3_arm_probabilities.pdf", bbox_inches="tight")
plt.savefig("plots/exp3_arm_probabilities.png", bbox_inches="tight")
plt.clf()
plt.cla()

# Plot regret
fig = plt.figure(figsize=(10, 10))
plt.plot(times, exp3_regret, label="EXP3", linewidth=4)
plt.plot(times, exp3s_regret, label="EXP3.S", linewidth=4)
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret: EXP3 vs EXP3S")
plt.legend()
plt.savefig("plots/exp3_regrets.pdf", bbox_inches="tight")
plt.savefig("plots/exp3_regrets.png", bbox_inches="tight")
