import numpy as np

# Implementing UCB as shown in https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/#mjx-eqn-equcb

class Ucb:
    def __init__(self, arm_means, delta):
        num_arms = len(arm_means)
        self.muStars = arm_means
        self.num_arms = num_arms
        self.T = np.zeros(num_arms)
        self.muHats = np.zeros(num_arms)
        self.regret = 0

    def pullArm(self, i):
        """Pull arm i"""
        #newReward = np.random.binomial(n=1, p=self.muStars[i])
        newReward = np.random.normal(loc=self.muStars[i], scale=1)
        totalReward = self.T[i] * self.muHats[i] + newReward
        self.muHats[i] = totalReward / (self.T[i] + 1)
        self.T[i] = self.T[i] + 1
        return newReward

    def t(self):
        """The index of the current round"""
        return sum(self.T) + 1

    def getUCBs(self):
        """Compute all the UCBs"""
        ucbs = np.zeros(self.num_arms)
        t = self.t()
        f_t = 1 + t * np.log(t)**2
        for i in range(self.num_arms):
            ucbs[i] = self.muHats[i] + np.sqrt(2*np.log(f_t)/self.T[i])
        return ucbs

    def sample(self):
        """Decide which arm to sample, then sample it"""
        # Make sure each arm has at least one sample
        if np.any(self.T == 0):
            reward = self.pullArm(i=np.where(self.T == 0)[0][0])
        else:
            # Otherwise, sample the one with the largest UCB
            ucbs = self.getUCBs()
            iStar = np.argmax(ucbs)
            reward = self.pullArm(i=iStar)

        self.regret += np.max(self.muStars) - reward

        return reward

    def switchArms(self, new_arm_means):
        """Change the arm means to new_arm_means, for the switching case"""
        self.muStars = new_arm_means
