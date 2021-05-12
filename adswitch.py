from SMPyBandits.Policies.AdSwitchNew import AdSwitchNew
import numpy as np

from tqdm import trange

# Setup
interval = 100
arm_rewards = {
    0: np.hstack((.7 * np.ones(interval), 0.3 * np.ones(interval), 0.5 * np.ones(interval), 0.3 * np.ones(interval))),
    1: np.hstack((.3 * np.ones(interval), 0.3 * np.ones(interval), 0.6 * np.ones(interval), 0.7 * np.ones(interval))),
    2: np.hstack((.5 * np.ones(interval), 0.7 * np.ones(interval), 0.3 * np.ones(interval), 0.4 * np.ones(interval))),
    3: np.hstack((.1 * np.ones(interval), 0.1 * np.ones(interval), 1.0 * np.ones(interval), 0.1 * np.ones(interval))),
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


algo = AdSwitchNew(N_ARMS, horizon=HORIZON, delta_s=4, delta_t=10)
algo.startGame()

starts = set()

S_sets = {}

for t in trange(HORIZON):
    noises = np.random.normal(0.0, scale=noise_std, size=N_ARMS)

    arm_idx = algo.choice()
    reward_at_idx = arm_rewards[arm_idx][t] + noises[arm_idx]

    algo.getReward(arm_idx, reward_at_idx)

    print(f"Episode start: {algo.start_of_episode}")
    starts.add(algo.start_of_episode)
    # if algo.start_of_episode == t:
    #     print("---------------------- HERE --------------------")
    if algo.set_S:
        S_sets[t] = algo.set_S.copy()

    


print(f"Starts: {starts}")