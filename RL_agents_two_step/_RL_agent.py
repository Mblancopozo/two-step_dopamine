import numpy as np
from numba import njit
from random import random, randint
import sys
import math
from scipy import stats

# -------------------------------------------------------------------------------------
# Utility functions.
# -------------------------------------------------------------------------------------
log_max_float = np.log(sys.float_info.max/2.1) # Log of largest possible floating point number.

def softmax(Q, T):
  "Softmax choice probs given values Q and inverse temp T."
  QT = Q * T
  QT[QT > log_max_float] = log_max_float  # Protection agairt overflow in exponential.
  expQT = np.exp(QT)
  return expQT / expQT.sum()

def array_softmax(Q, T):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    P = np.zeros(Q.shape)
    TdQ = -T * (Q[:, 0] - Q[:, 1])
    TdQ[TdQ > log_max_float] = log_max_float  # Protection against overflow in exponential.
    P[:, 0] = 1. / (1. + np.exp(TdQ))
    P[:, 1] = 1. - P[:, 0]
    return P

def array_softmax_flexible(Q_minus, T):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    P = np.zeros((len(Q_minus), 2))
    TdQ = -T * Q_minus
    TdQ[TdQ > log_max_float] = log_max_float  # Protection against overflow in exponential.
    P[:, 0] = 1. / (1. + np.exp(TdQ))
    P[:, 1] = 1. - P[:, 0]
    return P

def protected_log(x):
  'Return log of x protected against giving -inf for very small values of x.'
  return np.log(((1e-200) / 2) + (1 - (1e-200)) * x)

def lognormpdf(x, mu, sigma):
  '''
  adapted from https://github.com/bbabayan/RL_beliefstate/blob/master/lognormpdf.m

  return log likelihood of normal distribution
  '''
  return -0.5 * ((x - mu)/sigma) ** 2 - np.log((np.sqrt(2 * np.pi) * sigma))


def choose(P):
  "Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"
  return sum(np.cumsum(P) < random())

# def exp_mov_ave(choice_hist, tau=8., initValue=0.5):
#   'Exponential Moving average for 1d data.'
#   m = np.exp(-1. / tau)
#   i = 1 - m
#   if len(choice_hist) < 2:
#     mov_ave = initValue
#   else:
#     mov_ave = choice_hist[-2] * m + i * choice_hist[-1]
#   return mov_ave
class exp_mov_ave:
  # Exponential moving average class.
  def __init__(self, tau, init_value=0):
    self.tau = tau
    self.init_value = init_value
    self.reset()

  def reset(self, init_value=None, tau=None):
    if tau:
      self.tau = tau
    if init_value:
      self.init_value = init_value
    self.value = self.init_value
    self._m = math.exp(-1. / self.tau)
    self._i = 1 - self._m

  def update(self, sample):
    self.value = (self.value * self._m) + (self._i * sample)

class temp_decay:
  def __init__(self, init_temp, decay):
    self.init_temp = init_temp
    self.decay = decay
    self.reset()

  def reset(self):
    self.temp = self.init_temp

  def decay(self):
    self.temp = self.temp * self.decay

def session_log_likelihood(session, Q_net, iTemp=None, remove_neutral=False):
  'Evaluate session log likelihood given choices, action values and softmax temp. - only free-choice trials'
  Q_net_free_c = Q_net[session.trial_data['free_choice']]
  choices_free = session.trial_data['choices'][session.trial_data['free_choice']]
  if remove_neutral == True:
    # print('removing neutral blocks')
    block_type_free = session.blocks['trial_rew_state'][session.trial_data['free_choice']]
    Q_net_free_c = Q_net_free_c[block_type_free != 2]
    choices_free = choices_free[block_type_free != 2]
  choice_probs = array_softmax(Q_net_free_c, iTemp)
  session_log_likelihood = protected_log(
      choice_probs[np.arange(len(Q_net_free_c)), choices_free])
  session_log_likelihood = np.sum(session_log_likelihood)
  return session_log_likelihood

def session_log_likelihood_flexible(session, Q_kernel, dict_Q, params, multpersv, iTemp=None):
  'Evaluate session log likelihood given choices, action values and softmax temp. - only free-choice trials'
  alpha_uncorr_pos, alpha_uncorr_neg, alpha_mf_pos, alpha_mf_neg, \
  lbd, forget, \
  p_r, \
  weight_mf, weight_mb_uncorr, weight_mb_corr = params[:10]

  Q_k = Q_kernel[session.trial_data['free_choice']] # this are just the kernels
  # Q_values from each component
  Q_c = dict_Q['Q_c'][session.trial_data['free_choice']]
  Q_mb_uncorr = dict_Q['Q_mb_uncorr'][session.trial_data['free_choice']]
  Q_mb_corr = dict_Q['Q_mb_corr'][session.trial_data['free_choice']]
  choices_free = session.trial_data['choices'][session.trial_data['free_choice']]

  if multpersv[0]:
    Q_net = weight_mf * stats.zscore(Q_c[:, 0] - Q_c[:, 1]) + \
            weight_mb_uncorr * stats.zscore(Q_mb_uncorr[:, 0] - Q_mb_uncorr[:, 1]) + \
            weight_mb_corr * stats.zscore(Q_mb_corr[:, 0] - Q_mb_corr[:, 1]) + \
            (Q_k[:, 0] - Q_k[:, 1]) + \
            multpersv[0] * multpersv[1]
  else:
    Q_net = weight_mf * stats.zscore(Q_c[:, 0] - Q_c[:, 1]) + \
            weight_mb_uncorr * stats.zscore(Q_mb_uncorr[:, 0] - Q_mb_uncorr[:, 1]) + \
            weight_mb_corr * stats.zscore(Q_mb_corr[:, 0] - Q_mb_corr[:, 1]) + \
            (Q_k[:, 0] - Q_k[:, 1])

  choice_probs = array_softmax_flexible(Q_net, iTemp)
  session_log_likelihood = protected_log(
      choice_probs[np.arange(len(Q_net)), choices_free])
  session_log_likelihood = np.sum(session_log_likelihood)
  return session_log_likelihood

def session_log_likelihood_decay(session, Q_net, temperature):
  'Evaluate session log likelihood given choices, action values and softmax temp. - only free-choice trials'
  Q_net_free_c = Q_net[session.trial_data['free_choice']]
  temperature = temperature[session.trial_data['free_choice']]

  temperature[temperature > sys.float_info.max] = sys.float_info.max  # Protection against overflow
  temperature[temperature < sys.float_info.min] = sys.float_info.min  # Protection against divide by 0 in true_divide.
  itemperature = 1/temperature
  itemperature[itemperature < sys.float_info.min] = sys.float_info.min
  choice_probs = array_softmax(Q_net_free_c, itemperature)

  choices_free = session.trial_data['choices'][session.trial_data['free_choice']]
  session_log_likelihood = protected_log(
      choice_probs[np.arange(len(Q_net_free_c)), choices_free])
  session_log_likelihood = np.sum(session_log_likelihood)
  return session_log_likelihood

@njit
def _multitrial_perserveration_kernel(choices, alphamultipersv):
  multipersv = np.zeros(len(choices))
  multipersv[0] = 0.5
  for i, c in enumerate(choices[:-1]):
    multipersv[i + 1] = ((1 - alphamultipersv) * multipersv[i]) + (alphamultipersv * c)
  multipersv -= 0.5

  return multipersv

def _reward_rate(outcomes, tau=8, init_value=0.5):
  moving_average = exp_mov_ave(tau=tau, init_value=init_value)
  moving_average_session = []
  for o in outcomes:
    moving_average.update(o)
    moving_average_session.append(moving_average.value)

  return np.asarray(moving_average_session)

def _temperature_decay(session_rew_rate, temp_init, reward_threshold, decay):
  temp = temp_init
  all_temp = []
  for t_reward_rate in session_rew_rate:
    if t_reward_rate >= reward_threshold:
      temp = temp * decay
    else:
      temp = temp_init
    all_temp.append(temp)

  return np.asarray(all_temp)

def _trans_rew_kernel(choices, transitions_CR, outcomes, trans_rew1, trans_rew2):
  kernel = np.zeros(len(choices))
  for i in range(len(choices)):
    if transitions_CR[i] is True:
      if outcomes[i] == 1:
        kernel[i] = trans_rew1
      elif outcomes[i] == 0:
        kernel[i] = trans_rew2
    elif transitions_CR[i] is False:
      if outcomes[i] == 1:
        kernel[i] = - trans_rew1
      elif outcomes[i] == 0:
        kernel[i] = - trans_rew2
  return kernel

def _trans_kernel(choices, transitions_CR, trans_C, trans_R):
  kernel = np.zeros(len(choices))
  for i in range(len(choices)):
    if transitions_CR[i] is True:
      kernel[i] = trans_C

    elif transitions_CR[i] is False:
      kernel[i] = - trans_R

  return kernel

class MultitrialPerseveration:
  def __init__(self, tau=None, init_value=0., alpha=None):
    if alpha is None: alpha = 1 - np.exp(-1 / tau)
    self.alpha = alpha
    self.init_value = init_value
    self.reset()

  def reset(self, init_value=None):
    if init_value:
      self.init_value = init_value
    self.ave = self.init_value

  def update(self, sample):
    self.ave = ((1 - self.alpha) * self.ave) + (self.alpha * sample)

def apply_kernels(Q_pre, choices, transitions_CR, second_steps, outcomes, transition_type, param_names, params, kernels,
                  return_choice_mov=False):
  '''Apply modifier to entire sessions Q values due to kernels.'''
  if not kernels:
    return Q_pre, []
  p_names = param_names
  bias = params[p_names.index('bs')] if 'bs' in p_names else 0.
  persv = params[p_names.index('persv')] if 'persv' in p_names else 0.
  rew = params[p_names.index('rew')] if 'rew' in p_names else 0.
  rewonly = params[p_names.index('rewonly')] if 'rewonly' in p_names else 0.
  nonrewonly = params[p_names.index('nonrewonly')] if 'nonrewonly' in p_names else 0.
  multpersv = params[p_names.index('multpersv')] if 'multpersv' in p_names else 0.
  alphamultipersv = params[p_names.index('alphamultipersv')] if 'alphamultipersv' in p_names else 0.
  # initultipersv = params[p_names.index('initmultipersv')] if 'initmultipersv' in p_names else 0.
  multpersv_w = params[p_names.index('multpersv_w')] if 'multpersv_w' in p_names else 0.
  multpersv_slow = params[p_names.index('multpersv_slow')] if 'multpersv_slow' in p_names else 0.
  multpersv_fast = params[p_names.index('multpersv_fast')] if 'multpersv_fast' in p_names else 0.
  slow_x = params[p_names.index('slow_x')] if 'slow_x' in p_names else 0.
  alphamultipersv_sf = params[p_names.index('alphamultipersv_sf')] if 'alphamultipersv_sf' in p_names else 0.
  trans_rew1 = params[p_names.index('trans_rew1')] if 'trans_rew1' in p_names else 0.
  trans_rew2 = params[p_names.index('trans_rew2')] if 'trans_rew2' in p_names else 0.
  trans_C = params[p_names.index('trans_C')] if 'trans_C' in p_names else 0.
  trans_R = params[p_names.index('trans_R')] if 'trans_R' in p_names else 0.

  if 'w_rew_as_cue_sym' in p_names:  # Symmetric reward-as-cue strategy.
    w_rew_as_cue = params[p_names.index('w_rew_as_cue_sym')]  # reward-as-cue weight
    z_rew_as_cue = 0.5  # reward-as-cue symmetry paramter.
  elif 'w_rew_as_cue_asym' in p_names:  # Asymmetric reward-as-cue strategy.
    w_rew_as_cue = params[p_names.index('w_rew_as_cue_asym')]  # reward-as-cue weight
    z_rew_as_cue = params[p_names.index('z_rew_as_cue')]  # reward-as-cue symmetry paramter.
  elif 'w_rew_as_cue_pos' in p_names:
    w_rew_as_cue = params[p_names.index('w_rew_as_cue_pos')]  # reward-as-cue weight
    z_rew_as_cue = 1  # reward-as-cue symmetry paramter.
  else:
    w_rew_as_cue = 0
    z_rew_as_cue = 0

  kernel_Qs = np.zeros((len(choices), 2))
  kernel_Qs[:, 1] += bias
  kernel_Qs[1:, 1] += persv * (choices[:-1]-0.5)
  trans = (-1 if transition_type==1 else 1)
  kernel_Qs[1:, 1] += rew * (((second_steps[:-1] ^ outcomes[:-1])-0.5) * trans)
  kernel_Qs[1:, 1] += rewonly * np.array([x if out==1 else 0 for x, out in zip((((second_steps[:-1] ^ outcomes[:-1])-0.5) * trans), outcomes)])
  kernel_Qs[1:, 1] += nonrewonly * np.array([x if out==0 else 0 for x, out in zip((((second_steps[:-1] ^ outcomes[:-1])-0.5) * trans), outcomes)])

  if w_rew_as_cue:
    rew_as_cue_kernel = (((second_steps[:-1] ^ outcomes[:-1]) - 0.5) * trans)
    kernel_Qs[1:, 1] += w_rew_as_cue * np.asarray(
      [z_rew_as_cue * x if out == 1 else (1 - z_rew_as_cue) * x for x, out in zip(rew_as_cue_kernel, outcomes)])

  choice_mov_ave = _multitrial_perserveration_kernel(choices, alphamultipersv) if 'multpersv' in p_names else 0.
  kernel_Qs[:, 1] += multpersv * choice_mov_ave

  choice_mov_ave_fast = _multitrial_perserveration_kernel(choices, alphamultipersv_sf) if 'multpersv_fast' in p_names else 0.
  choice_mov_ave_slow = _multitrial_perserveration_kernel(choices, alphamultipersv_sf * slow_x) if 'multpersv_slow' in p_names else 0.
  kernel_Qs[:, 1] += multpersv_fast * choice_mov_ave_fast
  kernel_Qs[:, 1] += multpersv_slow * choice_mov_ave_slow

  trans_rew_kernel = _trans_rew_kernel(choices, transitions_CR, outcomes, trans_rew1, trans_rew2)
  kernel_Qs[:, 1] += trans_rew_kernel

  trans_kernel = _trans_kernel(choices, transitions_CR, trans_C, trans_R)
  kernel_Qs[:, 1] += trans_kernel

  if 'temp_init' in p_names:
    tau_rew_rate = params[p_names.index('tau_rew_rate')] # tau for exponential moving average reward rate
    # init_rew_rate = params[p_names.index('init_rew_rate')] # init value for exponential moving average of reward rate
    rew_threshold = params[p_names.index('rew_threshold')] #reward threshold to start decaying
    decay = params[p_names.index('decay')] # decay rate for temperature so it promotes exploitation
    temp_init = params[p_names.index('temp_init')] #initial temperature and reset temperature

    tau_rew_rate = tau_rew_rate if tau_rew_rate > 0 else sys.float_info.min
    session_reward_rate = _reward_rate(outcomes, tau=tau_rew_rate, init_value=0.5)
    temperature = _temperature_decay(session_reward_rate, temp_init=temp_init, reward_threshold=rew_threshold, decay=decay)
  else:
    # temperature = np.repeat([[1,1]], len(outcomes), axis=0)
    temperature = np.repeat(1, len(outcomes))

  if return_choice_mov:
    return Q_pre + kernel_Qs, temperature, choice_mov_ave
  else:
    return Q_pre + kernel_Qs, temperature

def init_kernels_sim(param_names, params):
  '''Initialise kernels at start of simulation run.'''
  p_names = param_names
  bias = params[p_names.index('bs')] if 'bs' in p_names else 0.
  persv = params[p_names.index('persv')] if 'persv' in p_names else 0.
  rew = params[p_names.index('rew')] if 'rew' in p_names else 0.
  rewonly = params[p_names.index('rewonly')] if 'rewonly' in p_names else 0.
  nonrewonly = params[p_names.index('nonrewonly')] if 'nonrewonly' in p_names else 0.
  alphamultipersv = params[p_names.index('alphamultipersv')] if 'alphamultipersv' in p_names else 0.
  # initultipersv = params[p_names.index('initultipersv')] if 'initultipersv' in p_names else 0.
  multpersv = params[p_names.index('multpersv')] if 'multpersv' in p_names else 0.
  multpersv_slow = params[p_names.index('multpersv_slow')] if 'multpersv_slow' in p_names else 0.
  multpersv_fast = params[p_names.index('multpersv_fast')] if 'multpersv_fast' in p_names else 0.
  slow_x = params[p_names.index('slow_x')] if 'slow_x' in p_names else 0.
  alphamultipersv_sf = params[p_names.index('alphamultipersv_sf')] if 'alphamultipersv_sf' in p_names else 0.

  if 'w_rew_as_cue_sym' in p_names:  # Symmetric reward-as-cue strategy.
    w_rew_as_cue = params[p_names.index('w_rew_as_cue_sym')]  # reward-as-cue weight
    z_rew_as_cue = 0.5  # reward-as-cue symmetry paramter.
  elif 'w_rew_as_cue_asym' in p_names:  # Asymmetric reward-as-cue strategy.
    w_rew_as_cue = params[p_names.index('w_rew_as_cue_asym')]  # reward-as-cue weight
    z_rew_as_cue = params[p_names.index('z_rew_as_cue')]  # reward-as-cue symmetry paramter.
  elif 'w_rew_as_cue_pos' in p_names:
    w_rew_as_cue = params[p_names.index('w_rew_as_cue_pos')]  # reward-as-cue weight
    z_rew_as_cue = 1  # reward-as-cue symmetry paramter.
  else:
    w_rew_as_cue = 0
    z_rew_as_cue = 0

  tau_rew_rate = params[p_names.index('tau_rew_rate')] if 'temp_init' in p_names else 0.# tau for exponential moving average reward rate
  # init_rew_rate = params[p_names.index('init_rew_rate')] if 'temp_init' in p_names else 0.# init value for exponential moving average of reward rate
  rew_threshold = params[p_names.index('rew_threshold')] if 'temp_init' in p_names else 0.#reward threshold to start decaying
  decay = params[p_names.index('decay')] if 'temp_init' in p_names else 0.# decay rate for temperature so it promotes exploitation
  temp_init = params[p_names.index('temp_init')] if 'temp_init' in p_names else 0.#initial temperature and reset temperature

  # if 'temp_init' in p_names:
  #   reward_mov_average = exp_mov_ave(tau=tau_rew_rate, init_value=init_rew_rate)
  #   temperature = temp_decay(init_temp=temp_init, decay=decay)
  # else:
  #   reward_mov_average = 0
  #   temperature = 0

  if 'multpersv' in p_names:
    multipersv_mov_ave = MultitrialPerseveration(alpha=alphamultipersv, init_value=0.5)
  else:
    multipersv_mov_ave = 0
    # return bias, persv, rew, rewonly, nonrewonly, multpersv, alphamultipersv, multipersv_mov_ave, \
    #        w_rew_as_cue, z_rew_as_cue
  if 'multpersv_fast' in p_names and not 'multpersv' in p_names:
    multipersv_mov_ave_fast = MultitrialPerseveration(alpha=alphamultipersv_sf, init_value=0.5)
    multipersv_mov_ave_slow = MultitrialPerseveration(alpha=alphamultipersv_sf * slow_x, init_value=0.5)
  else:
    multipersv_mov_ave_fast = 0
    multipersv_mov_ave_slow = 0
    # return bias, persv, rew, rewonly, nonrewonly, multpersv_slow, multpersv_fast, alphamultipersv_sf, multipersv_mov_ave_fast, multipersv_mov_ave_slow, \
    #        w_rew_as_cue, z_rew_as_cue

  # else:
  # return bias, persv, rew, rewonly, nonrewonly, multpersv, alphamultipersv, \
  #          w_rew_as_cue, z_rew_as_cue
  return bias, persv, rew, rewonly, nonrewonly, multpersv, multpersv_slow, multpersv_fast, alphamultipersv, alphamultipersv_sf, \
         slow_x, multipersv_mov_ave, multipersv_mov_ave_fast, multipersv_mov_ave_slow, w_rew_as_cue, z_rew_as_cue

def init_exploration_tradeoff_sim(param_names, params):
  p_names = param_names

  tau_rew_rate = params[p_names.index('tau_rew_rate')]  # tau for exponential moving average reward rate
  # init_rew_rate = params[p_names.index('init_rew_rate')]  # init value for exponential moving average of reward rate
  rew_threshold = params[p_names.index('rew_threshold')]  # reward threshold to start decaying
  decay = params[p_names.index('decay')]  # decay rate for temperature so it promotes exploitation
  temp_init = params[p_names.index('temp_init')]  # initial temperature and reset temperature

  reward_mov_average = exp_mov_ave(tau=tau_rew_rate, init_value=0.5)
  temperature = temp_decay(init_temp=temp_init, decay=decay)

  return rew_threshold, reward_mov_average, temperature

def apply_exploration_tradeoff_sim(o, kernel_param_values):
  rew_threshold, reward_mov_average, temperature = kernel_param_values

  reward_mov_average.update(o)

  if reward_mov_average.value >= rew_threshold:
    temperature.decay
  else:
    temperature.reset

  return temperature.temp


def apply_kernels_sim(Q_pre, c, ss, o, transition_type, param_names, kernel_param_values, kernels):
  ''' Evaluate modifier to action values due to kernels for single trial, called
          on each trials of simulation run.'''
  if not kernels:
    return Q_pre
  # if 'multpersv' in param_names:
  #   bias, persv, rew, rewonly, nonrewonly, multpersv, alphamultipersv, multipersv_mov_ave, \
  #   w_rew_as_cue, z_rew_as_cue = kernel_param_values
  # else:
  #   bias, persv, rew, rewonly, nonrewonly, multpersv, alphamultipersv, \
  #   w_rew_as_cue, z_rew_as_cue = kernel_param_values
  bias, persv, rew, rewonly, nonrewonly, multpersv, multpersv_slow, multpersv_fast, alphamultipersv, alphamultipersv_sf, \
  slow_x, multipersv_mov_ave, multipersv_mov_ave_fast, multipersv_mov_ave_slow, w_rew_as_cue, z_rew_as_cue = kernel_param_values

  kernel_Qs = np.zeros(2)
  kernel_Qs[1] += bias
  kernel_Qs[1] += persv * (c-0.5)
  trans = (-1 if transition_type==1 else 1)
  kernel_Qs[1] += rew * (((ss ^ o)-0.5) * trans)
  kernel_Qs[1] += rewonly * ((((ss ^ o)-0.5) * trans) if o==1 else 0)
  kernel_Qs[1] += nonrewonly * ((((ss ^ o)-0.5) * trans) if o==0 else 0)

  if w_rew_as_cue:
    rew_as_cue_kernel = (((ss ^ o)-0.5) * trans)
    kernel_Qs[1] += w_rew_as_cue * (z_rew_as_cue * rew_as_cue_kernel if o==1 else (1 - z_rew_as_cue) * rew_as_cue_kernel)

  if 'multpersv' in param_names:
    multipersv_mov_ave.update(c)
    kernel_Qs[1] += multpersv * (multipersv_mov_ave.ave - 0.5)

  if 'multpersv_fast' in param_names:
    multipersv_mov_ave_fast.update(c)
    multipersv_mov_ave_slow.update(c)
    kernel_Qs[1] += multpersv_fast * (multipersv_mov_ave_fast.ave - 0.5)
    kernel_Qs[1] += multpersv_slow * (multipersv_mov_ave_slow.ave - 0.5)

  # if 'expl_tradeoff' in param_names:
  #   reward_mov_average.update(o)
  #
  #   if reward_mov_average.value >= rew_threshold:
  #     temperature.update()
  #   else:
  #     temperature.reset()

  return Q_pre + kernel_Qs


def session_likelihood(agent, session, params, get_Qval=False):
  choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data('CTSO')
  n_trials = session.n_trials
  transition_type = session.blocks['transition_states'][0]
  transitions_CR = transitions_AB == session.blocks['trial_trans_state']  # Trial by trial common or rare transitions (True: common; False: rare)

  dict_Q = agent.value_func(choices, second_steps, outcomes, n_trials, transition_type, params)

  # Evaluate net action values and likelihood.
  Q_net, temperature = apply_kernels(dict_Q['Q_net'], choices, transitions_CR, second_steps, outcomes, transition_type,
                                     agent.param_names, params, agent.use_kernels)

  iTemp = (params[agent.param_names.index('iTemp')] if 'iTemp' in agent.param_names else 1)
  if get_Qval:
    dict_Q['Q_net'] = Q_net
    return dict_Q
  else:
    if 'temp_init' in agent.param_names:
      return session_log_likelihood_decay(session, Q_net, temperature)
    else:
      return session_log_likelihood(session, Q_net, iTemp, remove_neutral=agent.remove_neutral)

def session_likelihood_flexible(agent, session, params, get_Qval=False):
  choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data('CTSO')
  n_trials = session.n_trials
  transition_type = session.blocks['transition_states'][0]
  transitions_CR = transitions_AB == session.blocks[
    'trial_trans_state']  # Trial by trial common or rare transitions (True: common; False: rare)

  dict_Q = agent.value_func(choices, second_steps, outcomes, n_trials, transition_type, params)
  Q_net, temperature, choice_mov_ave = apply_kernels(dict_Q['Q_net'], choices, transitions_CR, second_steps, outcomes, transition_type,
                                     agent.param_names, params, agent.use_kernels, return_choice_mov=True) # here the Q_net is just the kernels

  iTemp = (params[agent.param_names.index('iTemp')] if 'iTemp' in agent.param_names else 1)
  multpersv_w = (params[agent.param_names.index('multpersv_w')] if 'multpersv_w' in agent.param_names else [])
  if get_Qval:
    dict_Q['Q_net'] = Q_net
    return dict_Q
  else:
    return session_log_likelihood_flexible(session, Q_net, dict_Q, params, [multpersv_w, choice_mov_ave], iTemp)


# -------------------------------------------------------------------------------------
# Base class
# -------------------------------------------------------------------------------------

class RL_agent:
  '''
  Kernel types:
          bs - Bias left vs right.
          persv - perseveration
          multpersv - multitrial perserveration
  '''

  def __init__(self, kernels=None):
    self.remove_neutral = False
    if kernels:
      self.use_kernels = True
      self.name = self.name + ''.join(['_' + k for k in kernels])
      for k in kernels:
        if k in ['bs', 'persv', 'rew', 'rewonly', 'nonrewonly', 'trans_rew1', 'trans_rew2', 'trans_C', 'trans_R']:
          self.param_names += [k]
          self.param_ranges += ['unc']
        elif k in ['multpersv']:
          self.param_names += [k]
          self.param_ranges += ['unc']
          self.param_names += ['alphamultipersv']
          self.param_ranges += ['unit']
        elif k in ['multpersv_w']:
          self.param_names += [k]
          self.param_ranges += ['pos']
          self.param_names += ['alphamultipersv']
          self.param_ranges += ['unit']
        elif k in ['multpersv_slow_fast']:
          self.param_names += ['multpersv_slow']
          self.param_ranges += ['unc']
          self.param_names += ['multpersv_fast']
          self.param_ranges += ['unc']
          self.param_names += ['alphamultipersv_sf']
          self.param_ranges += ['unit']
          self.param_names += ['slow_x']
          self.param_ranges += ['unit']
          # self.param_names += ['initultipersv']
          # self.param_ranges += ['unit']
        elif k in ['rew_as_cue_sym']:  # Symmetric reward-as-cue strategy.
          self.param_names += ['w_rew_as_cue_sym']  # Reward-as-cue weight.
          self.param_ranges += ['pos']
        elif k in ['rew_as_cue_asym']:  # Asymmetric reward-as-cue strategy.
          self.param_names += ['w_rew_as_cue_asym']  # Reward-as-cue weight.
          self.param_ranges += ['pos']
          self.param_names += ['z_rew_as_cue']  # Reward-as-cue asymmetry paramter.
          self.param_ranges += ['unit']
        elif k in ['rew_as_cue_pos']:  # Positive outcomes only reward-as-cue strategy.
          self.param_names += ['w_rew_as_cue_pos']  # Reward-as-cue weight.
          self.param_ranges += ['pos']
        elif k in ['expl_tradeoff']: #exploration decay with higher reward rate
          self.param_names += ['tau_rew_rate']
          self.param_ranges += ['pos']
          # self.param_names += ['init_rew_rate']
          # self.param_ranges += ['unit']
          self.param_names += ['rew_threshold']
          self.param_ranges += ['unit']
          self.param_names += ['decay']
          self.param_ranges += ['unit']
          self.param_names += ['temp_init']
          self.param_ranges += ['pos']
        elif k in ['remove_neutral']: #This is not a kernel, but it does not consider neutral blocks when computing log likelihood
          self.remove_neutral = True

        else:
          assert False, 'Kernel type not recognised.'
    else:
      self.use_kernels = False
    self.n_params = len(self.param_names)
    self.type = 'RL'

