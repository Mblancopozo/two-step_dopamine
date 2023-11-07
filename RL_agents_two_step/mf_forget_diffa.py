from ._RL_agent import *
from numba import njit

@njit
def mf_forget_diffa_session_value(choices, second_steps, outcomes, n_trials, transition_type, params):
  # Unpack parameters.
  alpha_pos, alpha_neg, weight_mf, lbd, forget = params[:5]

  # Variables.
  Q_c = np.zeros((n_trials, 2))  # Value of the choice.
  V_s = np.zeros((n_trials, 2))  # Value of the second-step.

  for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)):  # loop over trials.

    # Update action values.
    if o == 1:
      alpha = alpha_pos
    else:
      alpha = alpha_neg

    nc = 1 - c  # Not chosen action
    ns = 1 - s  # State not reached at second step.

    Q_c[i + 1, nc] = (1 - forget) * Q_c[i, nc] + forget * 0.5  # forgetting of the not chosen action, decay towards value is neutral
    V_s[i + 1, ns] = (1 - forget) * V_s[i, ns] + forget * 0.5  # forgetting of the no reached ss, decay towards value is neutral

    Q_c[i + 1, c] = (1. - alpha) * Q_c[i, c] + alpha * ((1. - lbd) * V_s[i, s] + lbd * o)  # First step TD update.
    V_s[i + 1, s] = (1. - alpha) * V_s[i, s] + alpha * o  # Second step TD update.

  # Evaluate net action values and likelihood
  Q_mf = weight_mf * Q_c

  dict_Q = {'Q_net': Q_mf,
            'Q_c': Q_c,
            'V_s': V_s}

  return dict_Q

def mf_forget_diffa_session_simulate(n_trials, task, param_names, params, kernels):
  # Unpack parameters.
  alpha_pos, alpha_neg, weight_mf, lbd, forget = params[:5]

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
    if o == 1:
      alpha = alpha_pos
    else:
      alpha = alpha_neg

    Q_c[i + 1, nc] = (1 - forget) * Q_c[
      i, nc] + forget * 0.5  # forgetting of the not chosen action, decay towards value is neutral
    V_s[i + 1, ns] = (1 - forget) * V_s[
      i, ns] + forget * 0.5  # forgetting of the no reached ss, decay towards value is neutral

    Q_c[i + 1, c] = (1. - alpha) * Q_c[i, c] + alpha * ((1. - lbd) * V_s[i, s] + lbd * o)  # First step TD update.
    V_s[i + 1, s] = (1. - alpha) * V_s[i, s] + alpha * o  # Second step TD update.

    Q_mf = weight_mf * Q_c[i+1, :]
    # Q_net = apply_kernels_sim(Q_c[i+1, :], c, s, o, task.blocks['transition_states'][0], param_names, kernel_param_values_init, kernels)
    Q_net = apply_kernels_sim(Q_mf, c, s, o, task.blocks['transition_states'][0], param_names, kernel_param_values_init, kernels)

  return choices, second_steps, outcomes, free_choice

class MF_forget_diffa(RL_agent):
  '''
  Model-free agent with asymmetric learning rates for rewarded and non-rewarded trials,
  and forgetting towards neutral value of the non-experienced second-step state
  '''

  def __init__(self, kernels=['bs', 'persv']):
    self.name = 'MF_forget_diffa'
    self.param_names = ['alpha_pos', 'alpha_neg', 'weight_mf', 'lbd', 'forget']
    self.param_ranges = ['unit', 'unit', 'pos', 'unit', 'unit']
    self.n_base_params = 5
    self.value_func = mf_forget_diffa_session_value
    RL_agent.__init__(self, kernels)

  def session_likelihood(self, session, params, get_Qval=False):
    return session_likelihood(self, session, params, get_Qval)

  def session_simulate(self, task, params, n_trials):
    return mf_forget_diffa_session_simulate(n_trials, task, self.param_names, params, self.use_kernels)
