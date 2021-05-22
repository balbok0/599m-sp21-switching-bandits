from SMPyBandits.Policies.AdSwitchNew import AdSwitchNew, Constant_C1, DELTA_T, DELTA_S
import numpy as np


class AdSwitchCustom(AdSwitchNew):
    def __init__(self, nbArms, horizon=None, C1=Constant_C1, delta_s=DELTA_S, delta_t=DELTA_T, *args, **kwargs):
        super().__init__(nbArms, horizon=horizon, C1=C1, delta_s=delta_s, delta_t=delta_t, *args, **kwargs)
        self.checked_good_arm_settings = set()

    def check_changes_good_arms(self):
        return self.check_changes_good_arms_new()

    def check_changes_good_arms_new(self):
        """ Check for changes of good arms.
        NOTE: Assumes that delta_s = 1 and delta_t = 1

        - I moved this into a function, in order to stop the 4 for loops (``good_arm``, ``s_1``, ``s_2``, ``s``) as soon as a change was detected (early stopping).
        - TODO this takes a crazy O(K t^3) time, it HAS to be done faster!
        """
        def subroutine(good_arm, s_1, s_2, s):
            # check condition (3)
            n_s1_s2_a = self.n_s_t(good_arm, s_1, s_2)  # sub interval [s1, s2] <= [s, t] (s <= s1 <= s2 <= t).
            mu_hat_s1_s2_a = self.mu_hat_s_t(good_arm, s_1, s_2)  # sub interval [s1, s2] <= [s, t] (s <= s1 <= s2 <= t).
            n_s_t_a = self.n_s_t(good_arm, s, self.t)
            mu_hat_s_t_a = self.mu_hat_s_t(good_arm, s, self.t)
            abs_difference_in_s1s2_st = abs(mu_hat_s1_s2_a - mu_hat_s_t_a)
            confidence_radius_s1s2 = np.sqrt(2 * max(1, np.log(self.horizon)) / max(n_s1_s2_a, 1))
            confidence_radius_st = np.sqrt(2 * max(1, np.log(self.horizon)) / max(n_s_t_a, 1))
            right_side = confidence_radius_s1s2 + confidence_radius_st
            # print("AdSwitchNew: should we start a new episode, by checking condition (3), with arm {}, s1 = {}, s2 = {}, s = {} and t = {}...".format(good_arm, s_1, s_2, s, self.t))  # DEBUG
            if abs_difference_in_s1s2_st > right_side:  # check condition 3:
                print("\n==> New episode was started, with arm {}, s1 = {}, s2 = {}, s = {} and t = {}, as condition (3) is satisfied!".format(good_arm, s_1, s_2, s, self.t))  # DEBUG
                # print("    n_s1_s2_a =", n_s1_s2_a)  # DEBUG
                # print("    mu_hat_s1_s2_a =", mu_hat_s1_s2_a)  # DEBUG
                # print("    n_s_t_a =", n_s_t_a)  # DEBUG
                # print("    mu_hat_s_t_a =", mu_hat_s_t_a)  # DEBUG
                # print("    abs_difference_in_s1s2_st =", abs_difference_in_s1s2_st)  # DEBUG
                # print("    confidence_radius_s1s2 =", confidence_radius_s1s2)  # DEBUG
                # print("    confidence_radius_st =", confidence_radius_st)  # DEBUG
                # print("    right_side =", right_side)  # DEBUG
                return True
            else:
                return False

        for good_arm in self.set_GOOD:
            for s_1 in range(self.start_of_episode, self.t + 1, self.delta_s):  # WARNING we could speed up this loop with their trick
                for s_2 in range(s_1, self.t + 1, self.delta_s):  # WARNING we could speed up this loop with their trick
                    # Only need to check s = self.t because all previous check were done before
                    if subroutine(good_arm, s_1, s_2, self.t):
                        return True

        # done for checking on good arms
        return False

    def check_changes_good_arms_old(self):
        """ Check for changes of good arms.

        - I moved this into a function, in order to stop the 4 for loops (``good_arm``, ``s_1``, ``s_2``, ``s``) as soon as a change was detected (early stopping).
        - TODO this takes a crazy O(K t^3) time, it HAS to be done faster!
        """
        for good_arm in self.set_GOOD:
            for s_1 in range(self.start_of_episode, self.t + 1, self.delta_s):  # WARNING we could speed up this loop with their trick
                for s_2 in range(s_1, self.t + 1, self.delta_s):  # WARNING we could speed up this loop with their trick
                    for s in range(self.start_of_episode, self.t + 1, self.delta_s):  # WARNING we could speed up this loop with their trick
                        setting = (good_arm, s_1, s_2, s)
                        if setting in self.checked_good_arm_settings:
                            continue
                        self.checked_good_arm_settings.add(setting)
                        # check condition (3)
                        n_s1_s2_a = self.n_s_t(good_arm, s_1, s_2)  # sub interval [s1, s2] <= [s, t] (s <= s1 <= s2 <= t).
                        mu_hat_s1_s2_a = self.mu_hat_s_t(good_arm, s_1, s_2)  # sub interval [s1, s2] <= [s, t] (s <= s1 <= s2 <= t).
                        n_s_t_a = self.n_s_t(good_arm, s, self.t)
                        mu_hat_s_t_a = self.mu_hat_s_t(good_arm, s, self.t)
                        abs_difference_in_s1s2_st = abs(mu_hat_s1_s2_a - mu_hat_s_t_a)
                        confidence_radius_s1s2 = np.sqrt(2 * max(1, np.log(self.horizon)) / max(n_s1_s2_a, 1))
                        confidence_radius_st = np.sqrt(2 * max(1, np.log(self.horizon)) / max(n_s_t_a, 1))
                        right_side = confidence_radius_s1s2 + confidence_radius_st
                        # print("AdSwitchNew: should we start a new episode, by checking condition (3), with arm {}, s1 = {}, s2 = {}, s = {} and t = {}...".format(good_arm, s_1, s_2, s, self.t))  # DEBUG
                        if abs_difference_in_s1s2_st > right_side:  # check condition 3:
                            print("\n==> New episode was started, with arm {}, s1 = {}, s2 = {}, s = {} and t = {}, as condition (3) is satisfied!".format(good_arm, s_1, s_2, s, self.t))  # DEBUG
                            # print("    n_s1_s2_a =", n_s1_s2_a)  # DEBUG
                            # print("    mu_hat_s1_s2_a =", mu_hat_s1_s2_a)  # DEBUG
                            # print("    n_s_t_a =", n_s_t_a)  # DEBUG
                            # print("    mu_hat_s_t_a =", mu_hat_s_t_a)  # DEBUG
                            # print("    abs_difference_in_s1s2_st =", abs_difference_in_s1s2_st)  # DEBUG
                            # print("    confidence_radius_s1s2 =", confidence_radius_s1s2)  # DEBUG
                            # print("    confidence_radius_st =", confidence_radius_st)  # DEBUG
                            # print("    right_side =", right_side)  # DEBUG
                            return True
        # done for checking on good arms
        return False