from ._RL_agent import *
from numba import njit


@njit
def latent_state_rewasym_session_value(choices, second_steps, outcomes, n_trials, transition_type, params):
  # Unpack parameters.
  p_r, weight_inf = params[:2]

  rew_good = 0.4
  rew_bad = 0.1
  non_rew = 0.5

  V_s = np.zeros((n_trials, 2))  # Second step TD values.

  p_o_1 = np.array([[non_rew, rew_bad],  # Probability of observed outcome given world in state 1.
                    [non_rew, rew_good]])  # Indices:  p_o_1[second_step, outcome]

  p_o_0 = np.array([[non_rew, rew_good],
                    [non_rew, rew_bad]])

  p_1 = np.zeros(n_trials)  # Probability world is in state 1.
  p_1[0] = 0.5

  if transition_type == 1:
    trans_prob = [0.8, 0.2]
  else:
    trans_prob = [0.2, 0.8]

  for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)):  # loop over trials.

    # Bayesian update of state probabilties given observed outcome.
    p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
    # Update of state probabilities due to possibility of block reversal.
    p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

    V1 = 0.8 * p_1[i + 1] + 0.2 * (1 - p_1[i + 1])

    V_s[i + 1, 0] = 1 - V1
    V_s[i + 1, 1] = V1

  Q_mb = trans_prob[0] * V_s + trans_prob[1] * V_s[:, ::-1]  # Model based action values.
  Q_mb = weight_inf * Q_mb

  dict_Q = {'Q_net': Q_mb,
            'Q_mb': Q_mb,
            'V_s': V_s}

  return dict_Q

def latent_state_rewasym_session_simulate(n_trials, task, param_names, params, kernels):
  # Unpack parameters.
  p_r, weight_inf = params[:2]

  rew_good = 0.4
  rew_bad = 0.1
  non_rew = 0.5

  V_s = np.zeros((n_trials + 1, 2))  # Second step TD values.


  Q_net = np.zeros(2)

  p_1 = 0.5  # Probability world is in state 1.

  p_o_1 = np.array([[non_rew, rew_bad],  # Probability of observed outcome given world in state 1.
                    [non_rew, rew_good]])  # Indices:  p_o_1[second_step, outcome]

  p_o_0 = np.array([[non_rew, rew_good],
                    [non_rew, rew_bad]])

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

    # Bayesian update of state probabilties given observed outcome.
    p_1 = p_o_1[s, o] * p_1 / (p_o_1[s, o] * p_1 + p_o_0[s, o] * (1 - p_1))
    # Update of state probabilities due to possibility of block reversal.
    p_1 = (1 - p_r) * p_1 + p_r * (1 - p_1)

    V1 = 0.8 * p_1 + 0.2 * (1 - p_1)

    V_s[i + 1, 0] = 1 - V1
    V_s[i + 1, 1] = V1

    Q_mb = trans_prob[0] * V_s[i + 1, :] + trans_prob[1] * V_s[i + 1, ::-1]  # Model based action values.
    Q_mb = weight_inf * Q_mb
    Q_net = apply_kernels_sim(Q_mb, c, s, o, task.blocks['transition_states'][0], param_names, kernel_param_values_init,
                              kernels)


  return choices, second_steps, outcomes, free_choice


class Latent_state_rewasym(RL_agent):
  '''
  Agent believes that there are two states of the world, but with an asymmetric effect of non-rewarded trials

  State 0, Second step 0 reward prob = rew_good, second step 1 reward prob = rew_bad, second step 0 & 1 non-reward prob = non_rew
  State 1, Second step 1 reward prob = rew_good, second step 0 reward prob = rew_bad, second step 0 & 1 non-reward prob = non_rew

  Agent believes the probability that the state of the world changes on each step is p_r.
  different after rewarded and unrewarded trials

  The agent infers which state of the world it is most likely to be in, and then chooses
  the action which leads to the best second step in that state
  '''

  def __init__(self, kernels=['bs', 'persv']):
    self.name = 'Latent_state_rewasym'
    self.param_names = ['p_r', 'weight_inf']
    self.param_ranges = ['unit', 'pos']
    self.n_base_params = 2
    self.value_func = latent_state_rewasym_session_value
    RL_agent.__init__(self, kernels)

  def session_likelihood(self, session, params, get_Qval=False):
    return session_likelihood(self, session, params, get_Qval)

  def session_simulate(self, task, params, n_trials):
    return latent_state_rewasym_session_simulate(n_trials, task, self.param_names, params, self.use_kernels)


