from ._RL_agent import *
from numba import njit

@njit
def mf_session_value(choices, second_steps, outcomes, n_trials, transition_type, params):
  # Unpack parameters.
  alpha, weight_mf, lbd = params[:3]

  # Variables.
  Q_c = np.zeros((n_trials, 2))  # Value of the choice.
  V_s = np.zeros((n_trials, 2))  # Value of the second-step.

  for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)):  # loop over trials.

    # Update action values.

    nc = 1 - c  # Not chosen action
    ns = 1 - s  # State not reached at second step.

    Q_c[i + 1, nc] = Q_c[i, nc]
    V_s[i + 1, ns] = V_s[i, ns]

    Q_c[i + 1, c] = (1. - alpha) * Q_c[i, c] + alpha * ((1. - lbd) * V_s[i, s] + lbd * o)  # First step TD update.
    V_s[i + 1, s] = (1. - alpha) * V_s[i, s] + alpha * o  # Second step TD update.

  # Evaluate net action values and likelihood
  Q_mf = weight_mf * Q_c

  dict_Q = {'Q_net': Q_mf,
            'Q_c': Q_c,
            'V_s': V_s}

  return dict_Q

def mf_session_simulate(n_trials, task, param_names, params, kernels):
  # Unpack parameters.
  alpha, weight_mf, lbd = params[:3]

  # Variables.
  Q_c = np.zeros((n_trials + 1, 2))  # Value of the choice.
  V_s = np.zeros((n_trials + 1, 2))  # Value of the second-step.
  Q_net = np.zeros(2)
  choices, second_steps, outcomes, free_choice = (np.zeros(n_trials, int), np.zeros(n_trials, int),
                                                  np.zeros(n_trials, int), np.zeros(n_trials, int))

  task.reset(n_trials)
  kernel_param_values_init = init_kernels_sim(param_names, params)

  for i in range(n_trials):
    # Generate trial events
    c = choose(softmax(Q_net, 1))
    c, s, o, fc = task.trial(c)
    choices[i], second_steps[i], outcomes[i], free_choice[i] = (c, s, o, fc)

    nc = 1 - c  # Not chosen action
    ns = 1 - s  # State not reached at second step

    # Update action values
    Q_c[i + 1, c] = (1. - alpha) * Q_c[i, c] + alpha * ((1. - lbd) * V_s[i, s] + lbd * o)  # First step TD update.
    V_s[i + 1, s] = (1. - alpha) * V_s[i, s] + alpha * o  # Second step TD update.

    Q_c[i + 1, nc] = Q_c[i, nc]
    V_s[i + 1, ns] = V_s[i, ns]

    Q_mf = weight_mf * Q_c[i+1, :]
    # Q_net = apply_kernels_sim(Q_c[i+1, :], c, s, o, task.blocks['transition_states'][0], param_names, kernel_param_values_init, kernels)
    Q_net = apply_kernels_sim(Q_mf, c, s, o, task.blocks['transition_states'][0], param_names, kernel_param_values_init, kernels)

  return choices, second_steps, outcomes, free_choice


class MF(RL_agent):
  '''
  Model-free agent
  '''

  def __init__(self, kernels=['bs', 'persv']):
    self.name = 'MF'
    self.param_names = ['alpha', 'weight_mf', 'lbd']
    self.param_ranges = ['unit', 'pos', 'unit']
    self.n_base_params = 3
    self.value_func = mf_session_value
    RL_agent.__init__(self, kernels)

  def session_likelihood(self, session, params, get_Qval=False):
    return session_likelihood(self, session, params, get_Qval)

  def session_simulate(self, task, params, n_trials):
    return mf_session_simulate(n_trials, task, self.param_names, params, self.use_kernels)


