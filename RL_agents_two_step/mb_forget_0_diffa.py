from ._RL_agent import *
from numba import njit


@njit
def mb_forget_0_diffa_session_value(choices, second_steps, outcomes, n_trials, transition_type, params):
  # Unpack parameters.
  alpha_pos, alpha_neg, weight_mb, forget = params[:4]

  if transition_type == 1:
    trans_prob = [0.8, 0.2]
  else:
    trans_prob = [0.2, 0.8]

  # Variables.
  V_s = np.zeros((n_trials, 2))  # Second step TD values.

  for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)):  # loop over trials.

    # Update action values and transition probabilities.
    if o == 1:
      alpha = alpha_pos
    else:
      alpha = alpha_neg

    ns = 1 - s  # State not reached at second step.

    V_s[i + 1, ns] = (1 - forget) * V_s[i, ns] # forgetting of the no reached ss, decay towards 0

    V_s[i + 1, s] = (1. - alpha) * V_s[i, s] + alpha * o  # Second step TD update.

  Q_mb = trans_prob[0] * V_s + trans_prob[1] * V_s[:, ::-1]  # Model based action values.
  Q_mb = weight_mb * Q_mb

  dict_Q = {'Q_net': Q_mb,
            'Q_mb': Q_mb,
            'V_s': V_s}

  return dict_Q

def mb_forget_0_diffa_session_simulate(n_trials, task, param_names, params, kernels):
  # Unpack parameters.
  alpha_pos, alpha_neg, weight_mb, forget = params[:4]

  # Variables.
  V_s = np.zeros((n_trials + 1, 2))  # Second step TD values.
  Q_net = np.zeros(2)
  choices, second_steps, outcomes, free_choice = (np.zeros(n_trials, int), np.zeros(n_trials, int),
                                                  np.zeros(n_trials, int), np.zeros(n_trials, int))

  task.reset(n_trials)
  if task.transition_block == 1:
    trans_prob = [0.8, 0.2]
  elif task.transition_block == 0:
    trans_prob = [0.2, 0.8]
  else:
    print('transition_block not being acquired properly')
  kernel_param_values_init = init_kernels_sim(param_names, params)

  for i in range(n_trials):
    # Generate trial events
    c = choose(softmax(Q_net, 1))
    c, s, o, fc = task.trial(c)
    choices[i], second_steps[i], outcomes[i], free_choice[i] = (c, s, o, fc)

    # Update action values and transition probabilities.
    if o == 1:
      alpha = alpha_pos
    else:
      alpha = alpha_neg
    ns = 1 - s  # State not reached at second step.

    V_s[i + 1, ns] = (1 - forget) * V_s[i, ns]   # forgetting of the no reached ss, decay towards 0

    V_s[i + 1, s] = (1. - alpha) * V_s[i, s] + alpha * o  # Second step TD update.

    Q_mb = trans_prob[0] * V_s[i+1, :] + trans_prob[1] * V_s[i+1, ::-1]  # Model based action values.
    Q_mb = weight_mb * Q_mb
    Q_net = apply_kernels_sim(Q_mb, c, s, o, task.blocks['transition_states'][0], param_names, kernel_param_values_init, kernels)

  return choices, second_steps, outcomes, free_choice



class MB_forget_0_diffa(RL_agent):
  '''
  Model-based agent with asymmetric learning rates for rewarded and non-rewarded trials,
  and forgetting towards 0 of the non-experienced second-step state
  '''

  def __init__(self, kernels=['bs', 'persv']):
    self.name = 'MB_forget_0_diffa'
    self.param_names = ['alpha_pos', 'alpha_neg', 'weight_mb', 'forget']
    self.param_ranges = ['unit', 'unit', 'pos', 'unit']
    self.n_base_params = 4
    self.value_func = mb_forget_0_diffa_session_value
    RL_agent.__init__(self, kernels)

  def session_likelihood(self, session, params, get_Qval=False):
    return session_likelihood(self, session, params, get_Qval)

  def session_simulate(self, task, params, n_trials):
    return mb_forget_0_diffa_session_simulate(n_trials, task, self.param_names, params, self.use_kernels)

