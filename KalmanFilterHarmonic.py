from numpy.linalg import inv
import numpy as np
from scipy.stats import multivariate_normal

class KalmanFilterHarmonic:

    def __init__(self, con, x, P, Q, R, start_index):
        self.x = np.array([[x]])
        self.P = P
        self.Q = Q
        self.R = R
        self.S = 0
        self.y = np.array([[0]])
        self.k = np.array([[0]])
        self.score = 0
        self.past_measurements = []
        self.past_states = []
        self.past_weights = []
        self.start_index = 0
        self.active = True
        self.confirmation_flag = False
        self.confirmation_observation_num = con
        self.num_past_predictions = 0
        self.num_past_updates = 0


    def prediction(self, F):
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + self.Q
        self.num_past_predictions += 1

        if len(self.past_states) <= self.confirmation_observation_num:
            if len(self.past_measurements) != len(self.past_states):
                self.kill()
        else:
            if self.confirmation_flag == False:
                self.confirmation_flag = True
           
        return

    def update(self, t, z, z_idx, h, r, w):

        self.S = h.dot(self.P).dot(h.T) + r
        self.k = self.P.dot(h.T.dot(inv(self.S)))
        self.y = z - h.dot(self.x)
        self.x = self.x + self.k.dot(self.y)
        self.P = self.P - self.k.dot(h.dot(self.P))

        self.past_measurements.append([t] + [z_idx])
        self.past_states.append(self.x)
        self.past_weights.append(w)

        self.score += np.mean(np.array([abs(x) for x in self.y]))
        self.num_past_updates += 1

        return

    def is_con(self):
        return self.confirmation_flag

    def get_err_variance(self):
        return self.P.item(0)

    def get_err_covariance(self):
        return self.P

    def get_inno_covariance(self):
        return self.S

    def set_err_covariance(self, P):
        self.P = P
        return

    def get_kalman_gain(self):
        return self.k

    def get_state(self):
        return self.x

    def push_state(self, x):
        self.past_states.append(x)
        self.past_weights.append(np.array([0]))
        return

    def get_post_fit_residual(self):
        return self.y
        # return self.k.dot(self.y).item

    def get_score(self, t):
        return self.score/(t+1-self.start_index)

    def get_past_measurements(self):
        return self.past_measurements

    def get_past_states(self):
        return self.past_states

    def get_past_weights(self):
        return self.past_weights

    def set_start_index(self, index):
        self.start_index = index

    def get_start_index(self):
        return self.start_index

    def get_end_index(self):
        return self.past_measurements[-1][0]

    def get_percentage_of_updates(self):
        return float(self.num_past_updates)/float(self.num_past_predictions)

    def false_alarm(self, error):
        self.score += error
        return

    def kill(self):
        self.active = False
        last_measurements_time_index = self.past_measurements[-1][0]
        self.past_states = self.past_states[0:last_measurements_time_index-(self.start_index-1)]   # remove every up to last measurement
        self.past_weights = self.past_weights[0:last_measurements_time_index - (self.start_index - 1)]   # remove every up to last measurement
        return

    def isactive(self):
        return self.active
