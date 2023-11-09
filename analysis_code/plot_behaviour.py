# -------------------------------------------------------------------------------------
# Code with functions to analyse and plot behaviour
# Marta Blanco-Pozo, 2023
# -------------------------------------------------------------------------------------

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from functools import partial
import pandas as pd
import statsmodels
from sklearn import linear_model as lm
import pingouin as pg
from matplotlib.collections import LineCollection

from Code_final_manuscript.code import parallel_processing as pp, plot_photometry as plp, \
  call_R_stats as R, import_regressors as ri, import_behaviour_data as di


def savefig(dir_folder, figname, svg=True, *args, **kwargs):
  plt.savefig(os.path.join(dir_folder, figname + '.pdf'), *args, **kwargs)
  if svg is True:
    plt.savefig(os.path.join(dir_folder, figname + '.svg'), *args, **kwargs)
  plt.close(plt.gcf())

## Exponential average choices
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
        self._m = math.exp(-1./self.tau)
        self._i = 1 - self._m

    def update(self, sample):
        self.value = (self.value * self._m) + (self._i * sample)

def _lag(x, i):  # Apply lag of i trials to array x.
  x_lag = np.zeros(x.shape, x.dtype)
  if i > 0:
      x_lag[i:] = x[:-i]
  else:
      x_lag[:i] = x[-i:]
  return x_lag

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None,
                              maxasterix=None):
  """
  from: https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
  Annotate barplot with p-values.

  :param num1: number of left bar to put bracket over
  :param num2: number of right bar to put bracket over
  :param data: string to write or number for generating asterixes
  :param center: centers of all bars (like plt.bar() input)
  :param height: heights of all bars (like plt.bar() input)
  :param yerr: yerrs of all bars (like plt.bar() input)
  :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
  :param barh: bar height in axes coordinates (0 to 1)
  :param fs: font size
  :param maxasterix: maximum number of asterixes to write (for very small p-values)
  """

  if type(data) is str:
    text = data
  else:
    # * is p < 0.05
    # ** is p < 0.005
    # *** is p < 0.0005
    # etc.
    text = ''
    p = .05

    while data < p:
      text += '*'
      p /= 10.

      if maxasterix and len(text) == maxasterix:
        break

    if len(text) == 0:
      text = 'n. s.'

  lx, ly = center[num1], height[num1]
  rx, ry = center[num2], height[num2]

  if yerr:
    ly += yerr[num1]
    ry += yerr[num2]

  ax_y0, ax_y1 = plt.gca().get_ylim()
  dh *= (ax_y1 - ax_y0)
  barh *= (ax_y1 - ax_y0)

  y = max(ly, ry) + dh

  barx = [lx, lx, rx, rx]
  bary = [y, y + barh, y + barh, y]
  mid = ((lx + rx) / 2, y + barh)

  plt.plot(barx, bary, c='black')

  kwargs = dict(ha='center', va='bottom')
  if fs is not None:
    kwargs['fontsize'] = fs

  plt.text(*mid, text, **kwargs)

def stats_annotation(p_val):
  return ['***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else '' for p in p_val]


def session_mov_average_attribute(session, tau=8, moving_average_variable='choices'):
  '''
  Return session's moving average of the variable in 'moving_average_variable'.
  e.g. choices moving average
  moving_average_variable: options: choices, second_steps, outcomes, transitions
  '''
  moving_average = exp_mov_ave(tau=tau, init_value=0.5)
  moving_average_session = []
  for x in session.trial_data[moving_average_variable]:
    moving_average.update(x)
    moving_average_session.append(moving_average.value)

  return np.asarray(moving_average_session)

def session_latencies(session, type):
  '''
  return latency between events
  type: 'start', 'choice', 'second-step', 'ITI', 'ITI-choice', 'ITI-start', 'ITI_poke_5', 'ITI_end_consump_init_trial'
  '''
  if type == 'start':
    times_e1, times_e2 = plp.get_times_consecutive_events(session, 'init_trial', 'poke_5', ['init_trial', 'poke_5'])
  elif type == 'choice':
    events = ['choice_state', 'choose_right', 'choose_left']
    all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                       if session.events[i].name in events])
    times_e1 = [session.events[x][0] for x in all_id[::2]]
    times_e2 = [session.events[x][0] for x in all_id[1::2]]
  elif type == 'second_step':
    events = ['up_state', 'down_state', 'choose_up', 'choose_down']
    all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                       if session.events[i].name in events])
    times_e1 = [session.events[x][0] for x in all_id[::2]]
    times_e2 = [session.events[x][0] for x in all_id[1::2]]

  elif type == 'second_step_poke':
    events = ['cue_up_state', 'poke_1']
    all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                       if session.events[i].name in events])
    times_e1_1 = [session.events[x][0] for x in all_id[::2]]
    times_e2_1 = [session.events[x][0] for x in all_id[1::2]]

    events = ['cue_down_state', 'poke_9']
    all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                       if session.events[i].name in events])
    times_e1_2 = [session.events[x][0] for x in all_id[::2]]
    times_e2_2 = [session.events[x][0] for x in all_id[1::2]]

    times_e1 = sorted(times_e1_1+times_e1_2)
    times_e2 = sorted(times_e2_1+times_e2_2)

  elif type == 'ITI':
    times_e1, times_e2 = plp.get_times_consecutive_events(session, 'inter_trial_interval', 'init_trial', ['inter_trial_interval', 'init_trial'])
  elif type == 'ITI-choice':
    events = ['inter_trial_interval', 'choose_right', 'choose_left']
    all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                 if session.events[i].name in events])
    times_e1 = [session.events[x][0] for x in all_id[::2]]
    times_e2 = [session.events[x][0] for x in all_id[1::2]]
  elif type == 'ITI-start':
    times_e1, times_e2 = plp.get_times_consecutive_events(session, 'inter_trial_interval', 'choice_state', ['inter_trial_interval', 'choice_state'])

  elif type == 'ITI_poke_5':
    times_e1, times_e2 = plp.get_times_consecutive_events(session, 'inter_trial_interval', 'poke_5', ['inter_trial_interval', 'poke_5'])

  elif type == 'ITI_end_consump_init_trial':
    times_e1, times_e2 = plp.get_times_consecutive_events(session, 'reward_consumption', 'choice_state', ['reward_consumption', 'choice_state'])


  latency = [e2 - e1 for e1, e2 in zip(times_e1, times_e2)][:len(times_e1)]

  return np.asarray(latency)

def session_correct(session):
  '''
  return array of 1,0, 0,5 for correct, incorrect or neutral trial
  '''
  return np.asarray([((~session.trial_data['choices'][i].astype(bool)).astype(int) ==
                      session.blocks['trial_rew_state'][i] ^ session.blocks['trial_trans_state'][i]) * 1 if
                     session.blocks['trial_rew_state'][i] != 2 else 0.5 for i in
                     range(len(session.trial_data['choices']))])


#%% Plotting functions
def plot_exp_mov_ave(session, save=False, simulated=False, moving_average_variable='choices', tau=8, ax=None,
                     plot_only_blocks=False, color_choices='b'):
  '''
  plot the exponential moving average in a session of the variable defined in moving_average_variable
  save = [dir_folder, save_name]
  '''
  ax = ax or plt.gca()

  moving_average = exp_mov_ave(tau=tau, init_value=0.5)
  moving_average_session = []
  for x in session.trial_data[moving_average_variable]:
    moving_average.update(x)
    moving_average_session.append(moving_average.value)

  end_block1 = np.where(session.blocks['trial_rew_state'][:-1] != session.blocks['trial_rew_state'][1:])[0]
  end_block2 = np.where(session.blocks['trial_trans_state'][:-1] != session.blocks['trial_trans_state'][1:])[0]
  pos_end_blocks = sorted(
    set(np.concatenate((end_block1, end_block2, [-1], [len(session.blocks['trial_rew_state'])]))))

  color = []
  bar_pos = []
  for i in range(len(session.blocks['transition_states'])):
    if session.blocks['transition_states'][i] == 1:
      color.append('orange')
      if session.blocks['reward_states'][i] == 1:
        bar_pos.append(0.75)
      elif session.blocks['reward_states'][i] == 0:
        bar_pos.append(0.25)
      else:
        bar_pos.append(0.5)
    else:
      color.append('blue')
      if session.blocks['reward_states'][i] == 1:
        bar_pos.append(0.25)
      elif session.blocks['reward_states'][i] == 0:
        bar_pos.append(0.75)
      else:
        bar_pos.append(0.5)

  mouse = session.subject_ID
  if plot_only_blocks == False:
    ax.plot(moving_average_session, color=color_choices)
  for i in range(len(pos_end_blocks) - 1):
    ax.plot((pos_end_blocks[i] + 1, pos_end_blocks[i + 1]), (bar_pos[i], bar_pos[i]), color=color[i])
  custom_lines = [plt.Line2D((0, 1), (0, 0), color='orange', lw=4),
                  plt.Line2D((0, 1), (0, 0), color='blue', lw=4)]
  ax.legend(custom_lines, ['Transition A', 'Transition B'])
  if session.reward_loc == 'UD':
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['right', '50%', 'left'])
  else:
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['down', '', 'up'])

  if plot_only_blocks == False:
    ax.set_ylabel('Choice moving\naverage')
    ax.set_xlabel('Trial #')
  if simulated == False:
    date = session.datetime_string
    version_stage = session.trial_data['stage'][0]
    ax.set_title('m{} - {} - stage{}'.format(mouse, date, version_stage))
  else:
    ax.set_title('simulated m{}'.format(mouse))

  plt.gcf().set_tight_layout(True)
  if save:
    savefig(save[0], '{}_m{} - {} - stage{}'.format(save[1], mouse, date, version_stage))

def plot_blocks(session, ax=None):
  '''
  Plot the probability of reward for each second-step port across a session
  '''
  ax = ax or plt.gca()
  rew_up = [0.8 if x == 1 else 0.2 if x == 0 else 0.5 for x in session.blocks['trial_rew_state']]
  rew_down = [0.8 if x == 0 else 0.2 if x == 1 else 0.5 for x in session.blocks['trial_rew_state']]
  ax.plot(rew_up, color='deepskyblue', label='up')
  ax.plot(rew_down, color='red', label='down')
  ax.set_yticks([0.2, 0.5, 0.8])
  ax.set_ylabel('Reward\nprobability')
  ax.set_xlabel('Trial#')
  ax.legend()
  plt.gcf().set_tight_layout(True)

def _compute_stay_probability_same_diff_sessions(sessions_sub, selection_type, select_n, block_type, forced_choice=False,
                                      correct_select=False, return_stay_prob_joint_trials=False):
  '''
  Compute stay probability looking at whether previous trials was same or different as the current trial.
  If return_stay_prob_joint_trials is False, return stay probability per session, and return mean across sessions.
  If return_stay_prob_joint_trials is True, return stay probability across all trials
  from all sessions
  '''
  stay_probability = np.array([])
  all_rew_same = np.array([])
  all_rew_diff = np.array([])
  all_nonrew_same = np.array([])
  all_nonrew_diff = np.array([])

  for session in sessions_sub:
    if len(session.blocks['trial_rew_state']) > 11:  # Analise if there were more than 10 trials

      positions = session.select_trials(selection_type=selection_type, select_n=select_n, block_type=block_type)

      choices = session.trial_data['choices']
      outcomes = session.trial_data['outcomes']
      transitions = session.trial_data['transitions']
      transition_type = session.blocks['trial_trans_state']
      free_choice_trials = session.trial_data['free_choice']
      rew_state = session.blocks['trial_rew_state']

      # Positions of trial type
      rew_same = np.where((choices[:-1] == _lag(choices, 1)[:-1]) & outcomes[:-1])[0]
      rew_diff = np.where(~(choices[:-1] == _lag(choices, 1)[:-1]) & outcomes[:-1])[0]
      nonrew_same = np.where((choices[:-1] == _lag(choices, 1)[:-1]) & ~outcomes[:-1])[0]
      nonrew_diff = np.where(~(choices[:-1] == _lag(choices, 1)[:-1]) & ~outcomes[:-1])[0]

      if correct_select == True:
        correct = ((~choices.astype(bool)).astype(int) == (transition_type ^ rew_state))

      if forced_choice == False:
        # select only block_type trials and Eliminate forced choice trials
        rew_same = [x for x in rew_same if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        rew_diff = [x for x in rew_diff if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_same = [x for x in nonrew_same if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_diff = [x for x in nonrew_diff if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
      else:
        # select only forced choice trials
        rew_same = [x for x in rew_same if
                      ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]
        rew_diff = [x for x in rew_diff if
                    ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_same = [x for x in nonrew_same if
                         ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_diff = [x for x in nonrew_diff if
                       ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]

      # Get probability of stay
      stay = (choices[1:] == choices[:-1]).astype(int)
      if return_stay_prob_joint_trials == True:
        all_rew_same = np.hstack((all_rew_same, stay[rew_same])) if all_rew_same.size else stay[rew_same]
        all_rew_diff = np.hstack((all_rew_diff, stay[rew_diff])) if all_rew_diff.size else stay[rew_diff]
        all_nonrew_same = np.hstack((all_nonrew_same, stay[nonrew_same])) if all_nonrew_same.size else stay[nonrew_same]
        all_nonrew_diff = np.hstack((all_nonrew_diff, stay[nonrew_diff])) if all_nonrew_diff.size else stay[nonrew_diff]

      else:
        stay_probs = np.zeros(4)
        stay_probs[0] = np.nanmean(stay[rew_same])  # Rewarded, common transition.
        stay_probs[1] = np.nanmean(stay[rew_diff])  # Rewarded, rare transition.
        stay_probs[2] = np.nanmean(stay[nonrew_same])  # Non-rewarded, common transition.
        stay_probs[3] = np.nanmean(stay[nonrew_diff])  # Non-rewarded, rare transition.

        stay_probability = np.vstack([stay_probability, stay_probs]) if stay_probability.size else stay_probs

  if return_stay_prob_joint_trials == True:
    stay_probs = np.zeros(4)
    stay_probs[0] = np.nanmean(all_rew_same)  # Rewarded, common transition.
    stay_probs[1] = np.nanmean(all_rew_diff)  # Rewarded, rare transition.
    stay_probs[2] = np.nanmean(all_nonrew_same)  # Non-rewarded, common transition.
    stay_probs[3] = np.nanmean(all_nonrew_diff)  # Non-rewarded, rare transition.

    return stay_probs, len(sessions_sub), stay_probs

  else:
    stay_probability_mean_sub = np.nanmean(stay_probability, axis=0) if len(
      stay_probability.shape) > 1 else stay_probability
    num_session_sub = len(stay_probability)

    return stay_probability_mean_sub, num_session_sub, stay_probability

def _compute_stay_probability_sessions(sessions_sub, selection_type, select_n, block_type, forced_choice=False,
                                       count_trial_types=False, correct_select=False, stim_select=False,
                                       return_stay_prob_joint_trials=False, same_prob_stim=False):
  '''
    Compute stay probability as function above, but using choices, left vs right. As in Akam, Costa & Dayan (2015)
    If return_stay_prob_joint_trials is False, return stay probability per session, and return mean across sessions.
    If return_stay_prob_joint_trials is True, return stay probability across all trials from all sessions
    '''
  stay_probability = np.array([])
  all_rew_common = np.array([])
  all_rew_rare = np.array([])
  all_nonrew_common = np.array([])
  all_nonrew_rare = np.array([])

  for session in sessions_sub:
    if len(session.blocks['trial_rew_state']) > 11:  # Analise if there were more than 10 trials

      positions = session.select_trials(selection_type=selection_type, select_n=select_n, block_type=block_type)

      choices = session.trial_data['choices']
      outcomes = session.trial_data['outcomes']
      transitions = session.trial_data['transitions']
      transition_type = session.blocks['trial_trans_state']
      free_choice_trials = session.trial_data['free_choice']
      rew_state = session.blocks['trial_rew_state']

      # Positions of trial type
      rew_common = np.where((transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
      rew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
      nonrew_common = np.where((transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
      nonrew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]

      if correct_select == True:
        correct = ((~choices.astype(bool)).astype(int) == (transition_type ^ rew_state))

      if forced_choice == False:
        # select only block_type trials and Eliminate forced choice trials
        rew_common = [x for x in rew_common if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
      elif forced_choice == 'short_ITI':
        # select only block_type trials and Eliminate forced choice trials and select trials after short ITI
        latency_ITI = session_latencies(session, 'ITI')
        latency_ITI = _lag(latency_ITI, 1)

        rew_common = [x for x in rew_common if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                       (x in np.where(latency_ITI < 2500)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                     (x in np.where(latency_ITI < 2500)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                          and (x in np.where(latency_ITI < 2500)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                        and (x in np.where(latency_ITI < 2500)[0]))]

      elif forced_choice == 'long_ITI':
        # select only block_type trials and Eliminate forced choice trials and select trials after short ITI
        latency_ITI = session_latencies(session, 'ITI')
        latency_ITI = _lag(latency_ITI, 1)

        rew_common = [x for x in rew_common if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                       (x in np.where(latency_ITI >= 3500)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                     (x in np.where(latency_ITI >= 3500)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                          and (x in np.where(latency_ITI >= 3500)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                        and (x in np.where(latency_ITI >= 3500)[0]))]
      elif forced_choice == 'short_choice':
        # select only block_type trials and Eliminate forced choice trials and select trials after short ITI
        latency_choice = session_latencies(session, 'choice')

        rew_common = [x for x in rew_common if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                       (x in np.where(latency_choice < 500)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                     (x in np.where(latency_choice < 500)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                          and (x in np.where(latency_choice < 500)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                        and (x in np.where(latency_choice < 500)[0]))]

      elif forced_choice == 'long_choice':
        # select only block_type trials and Eliminate forced choice trials and select trials after short ITI
        latency_choice = session_latencies(session, 'choice')

        rew_common = [x for x in rew_common if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                       (x in np.where(latency_choice >= 500)[0]) and (x in np.where(latency_choice < 3000)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                     (x in np.where(latency_choice >= 500)[0]) and (x in np.where(latency_choice < 3000)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                          and (x in np.where(latency_choice >= 500)[0]) and (x in np.where(latency_choice < 3000)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                        and (x in np.where(latency_choice >= 500)[0]) and (x in np.where(latency_choice < 3000)[0]))]
      elif forced_choice == 'short_ITI_choice':
        # select only block_type trials and Eliminate forced choice trials and select trials after short ITI
        latency_choice = session_latencies(session, 'ITI-choice')
        latency_choice = _lag(latency_choice, 1)

        rew_common = [x for x in rew_common if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                       (x in np.where(latency_choice < 4300)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                     (x in np.where(latency_choice < 4300)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                          and (x in np.where(latency_choice < 4300)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                        and (x in np.where(latency_choice < 4300)[0]))]

      elif forced_choice == 'long_ITI_choice':
        # select only block_type trials and Eliminate forced choice trials and select trials after short ITI
        latency_choice = session_latencies(session, 'ITI-choice')
        latency_choice = _lag(latency_choice, 1)

        rew_common = [x for x in rew_common if
                      ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                       (x in np.where(latency_choice >= 7000)[0]) and (x in np.where(latency_choice < 1000000)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]) and
                     (x in np.where(latency_choice >= 7000)[0]) and (x in np.where(latency_choice < 1000000)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                          and (x in np.where(latency_choice >= 7000)[0]) and (x in np.where(latency_choice < 1000000)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0])
                        and (x in np.where(latency_choice >= 7000)[0]) and (x in np.where(latency_choice < 1000000)[0]))]
      else:
        rew_common = [x for x in rew_common if
                      ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x in np.where(free_choice_trials == False)[0]) and (x in np.where(positions == True)[0]))]

      if stim_select == True:
        # select_after stim
        stim = session.trial_data['stim']
        rew_common = [x for x in rew_common if (x in np.where(stim == 1)[0])]
        rew_rare = [x for x in rew_rare if (x in np.where(stim == 1)[0])]
        nonrew_common = [x for x in nonrew_common if (x in np.where(stim == 1)[0])]
        nonrew_rare = [x for x in nonrew_rare if (x in np.where(stim == 1)[0])]

      elif stim_select == 'non_stim':
        # select after non_stim trials
        stim = session.trial_data['stim']
        rew_common = [x for x in rew_common if (x in np.where(stim == 0)[0])]
        rew_rare = [x for x in rew_rare if (x in np.where(stim == 0)[0])]
        nonrew_common = [x for x in nonrew_common if (x in np.where(stim == 0)[0])]
        nonrew_rare = [x for x in nonrew_rare if (x in np.where(stim == 0)[0])]

      if same_prob_stim == True:
        stim = session.trial_data['stim']
        rew_common = [x for x in rew_common if (x - 1 not in np.where(stim == True)[0]) and (x - 2 not in np.where(stim == True)[0])]
        rew_rare = [x for x in rew_rare if (x - 1 not in np.where(stim == True)[0]) and (x - 2 not in np.where(stim == True)[0])]
        nonrew_common = [x for x in nonrew_common if (x - 1 not in np.where(stim == True)[0]) and (x - 2 not in np.where(stim == True)[0])]
        nonrew_rare = [x for x in nonrew_rare if (x - 1 not in np.where(stim == True)[0]) and (x - 2 not in np.where(stim == True)[0])]

      if count_trial_types == True:
        if correct_select == True:
          count = np.zeros(12)
          correct_id = np.where(correct == True)[0]
          neutral_id = np.where(rew_state == 2)[0]
          incorrect_id = np.array([x for x in np.where(correct == False)[0] if x not in neutral_id])

          correct_rew_common = [x for x in rew_common if x in correct_id]
          correct_rew_rare = [x for x in rew_rare if x in correct_id]
          correct_nonrew_common = [x for x in nonrew_common if x in correct_id]
          correct_nonrew_rare = [x for x in nonrew_rare if x in correct_id]
          incorrect_rew_common = [x for x in rew_common if x in incorrect_id]
          incorrect_rew_rare = [x for x in rew_rare if x in incorrect_id]
          incorrect_nonrew_common = [x for x in nonrew_common if x in incorrect_id]
          incorrect_nonrew_rare = [x for x in nonrew_rare if x in incorrect_id]
          neutral_rew_common = [x for x in rew_common if x in neutral_id]
          neutral_rew_rare = [x for x in rew_rare if x in neutral_id]
          neutral_nonrew_common = [x for x in nonrew_common if x in neutral_id]
          neutral_nonrew_rare = [x for x in nonrew_rare if x in neutral_id]

          count[0] = len(correct_rew_common)
          count[1] = len(correct_rew_rare)
          count[2] = len(correct_nonrew_common)
          count[3] = len(correct_nonrew_rare)
          count[4] = len(incorrect_rew_common)
          count[5] = len(incorrect_rew_rare)
          count[6] = len(incorrect_nonrew_common)
          count[7] = len(incorrect_nonrew_rare)
          count[8] = len(neutral_rew_common)
          count[9] = len(neutral_rew_rare)
          count[10] = len(neutral_nonrew_common)
          count[11] = len(neutral_nonrew_rare)

          stay_probability = np.vstack([stay_probability, count]) if stay_probability.size else count
        else:
          count = np.zeros(4)
          count[0] = len(rew_common)
          count[1] = len(rew_rare)
          count[2] = len(nonrew_common)
          count[3] = len(nonrew_rare)

          stay_probability = np.vstack([stay_probability, count]) if stay_probability.size else count

      else:
        # Get probability of stay
        stay = (choices[1:] == choices[:-1]).astype(int)
        if return_stay_prob_joint_trials == True:
          all_rew_common = np.hstack((all_rew_common, stay[rew_common])) if all_rew_common.size else stay[rew_common]
          all_rew_rare = np.hstack((all_rew_rare, stay[rew_rare])) if all_rew_rare.size else stay[rew_rare]
          all_nonrew_common = np.hstack((all_nonrew_common, stay[nonrew_common])) if all_nonrew_common.size else stay[nonrew_common]
          all_nonrew_rare = np.hstack((all_nonrew_rare, stay[nonrew_rare])) if all_nonrew_rare.size else stay[nonrew_rare]

        else:
          stay_probs = np.zeros(4)
          stay_probs[0] = np.nanmean(stay[rew_common])  # Rewarded, common transition.
          stay_probs[1] = np.nanmean(stay[rew_rare])  # Rewarded, rare transition.
          stay_probs[2] = np.nanmean(stay[nonrew_common])  # Non-rewarded, common transition.
          stay_probs[3] = np.nanmean(stay[nonrew_rare])  # Non-rewarded, rare transition.

          stay_probability = np.vstack([stay_probability, stay_probs]) if stay_probability.size else stay_probs

  if count_trial_types == True:
    return stay_probability

  elif return_stay_prob_joint_trials == True:
    stay_probs = np.zeros(4)
    stay_probs[0] = np.nanmean(all_rew_common)  # Rewarded, common transition.
    stay_probs[1] = np.nanmean(all_rew_rare)  # Rewarded, rare transition.
    stay_probs[2] = np.nanmean(all_nonrew_common)  # Non-rewarded, common transition.
    stay_probs[3] = np.nanmean(all_nonrew_rare)  # Non-rewarded, rare transition.

    return stay_probs, len(sessions_sub), stay_probs

  else:
    stay_probability_mean_sub = np.nanmean(stay_probability, axis=0) if len(
      stay_probability.shape) > 1 else stay_probability
    num_session_sub = len(stay_probability)

    return stay_probability_mean_sub, num_session_sub, stay_probability


def compute_stay_probability(sessions, initial_trials, reward_type='all', selection_type='end', forced_choice=False,
                             return_subjects=False, subjects=[], stim_select=False, return_stay_prob_joint_trials=False,
                             same_prob_stim=False, stay_same_diff=False):
  '''
  Compute stay probability across subjects
  reward_type: 'all', 'neutral', 'non_neutral'
  initial_trials: number of trials to eliminate after reversal
  return_stay_prob_joint_trials: False, compute stay probability per each session separately, else if True, concatenate
    all trials and compute stay probability across all trials per animal
  stay_same_diff: if True, compute stay probability using whether current trial is same or different as previous trial;
    else if False, use actual choices (left choice as reference) - As in Akam, Costa & Dayan (2015)
  '''
  stay_probability_all_means = np.array([])
  num_session_sub_all = []
  sessions_sub = []
  if subjects == []:
    subjects = list(set([sessions[i].subject_ID for i in range(len(sessions))]))
  for sub in subjects:
    idx_sub = np.where([sessions[i].subject_ID == sub for i in range(len(sessions))])[0]
    sessions_sub.append([sessions[x] for x in idx_sub])

  if stay_same_diff:
    stay_prob_compute = pp.map(partial(_compute_stay_probability_same_diff_sessions,
                                       selection_type=selection_type, select_n=initial_trials, block_type=reward_type,
                                       forced_choice=forced_choice,
                                      return_stay_prob_joint_trials=return_stay_prob_joint_trials), sessions_sub)
  else:
    stay_prob_compute = pp.map(partial(_compute_stay_probability_sessions,selection_type=selection_type,
                                       select_n=initial_trials, block_type=reward_type, forced_choice=forced_choice,
                                       stim_select=stim_select, return_stay_prob_joint_trials=return_stay_prob_joint_trials,
                                       same_prob_stim=same_prob_stim), sessions_sub)
  stay_probability_all_means, num_session_sub_all, stay_prob_session_all = zip(*stay_prob_compute)
  stay_probability_all_means = np.vstack(stay_probability_all_means)

  stay_probability_mean_all = np.nanmean(stay_probability_all_means, axis=0) \
    if len(stay_probability_all_means.shape) > 1 else stay_probability_all_means
  stay_probability_sem_all = stats.sem(stay_probability_all_means, axis=0, nan_policy='omit') \
    if len(stay_probability_all_means.shape) > 1 else 0
  sum_num_sessions = np.sum(num_session_sub_all)

  if return_subjects == True:
    return (stay_probability_all_means, stay_probability_mean_all, stay_probability_sem_all, sum_num_sessions, stay_prob_session_all, subjects)
  else:
    return (stay_probability_all_means, stay_probability_mean_all, stay_probability_sem_all, sum_num_sessions, stay_prob_session_all)


def plot_stay_probability(stay_probability_mean, stay_probability_sem, subjects, stay_prob_session, fontsize=20,
                          stay_probability_per_session=False, errorbar=False, scatter=False, plot_significance=False,
                          stats=True, ax=None, stay_same_diff=False):
  '''
  Plot stay probability
  '''
  ax = ax or plt.gca()
  colors = ['orange', 'blue', 'orange', 'blue']
  ax.bar(np.arange(1, 5), stay_probability_mean, yerr=stay_probability_sem,
          error_kw={'ecolor': 'k', 'capsize': 3, 'elinewidth': 2, 'markeredgewidth': 2}, color=colors,edgecolor=colors, linewidth=1, fill=True, zorder=-1)
  if stay_probability_per_session != False:
    y = np.vstack([np.nanmean(s, axis=0) if s.ndim > 1 else s for s in stay_probability_per_session])
    sem = np.vstack([stats.sem(s, axis=0, nan_policy='omit') if s.ndim > 1 else [np.nan, np.nan, np.nan, np.nan] for s in stay_probability_per_session])
    x = np.random.normal(0, 0.14, size=len(stay_probability_per_session))
    for i in np.arange(1, 5):
      if errorbar == True:
        [ax.errorbar(x[e] + i, y.T[i-1][e], sem.T[i-1][e], c=colors[i-1], fmt='o', ms=2) for e in range(len(y))]
      elif scatter == True:
        ax.scatter(x+i, y.T[i-1], c='dimgray', s=5, lw=0, zorder=1, alpha=0.7)

  if stay_same_diff:
    names = ['same', 'diff', 'same', 'diff']
    name_variable = 'same/diff'
  else:
    names = ['common', 'rare', 'common', 'rare']
    name_variable = 'Transition'
  if stats == True:
    subjects_column = np.hstack(
      [np.repeat('{}'.format(sub), 4 * len(stay_prob_session[i])) if stay_prob_session[i].ndim > 1
       else np.repeat('{}'.format(sub), 4) for i, sub in enumerate(subjects)])
    transition_column = np.hstack(
      [names * len(stay_prob_session[i]) if stay_prob_session[i].ndim > 1
       else names for i in range(len(stay_prob_session))])
    outcome_column = np.hstack(
      [['rew', 'rew', 'unrew', 'unrew'] * len(stay_prob_session[i]) if stay_prob_session[i].ndim > 1
       else ['rew', 'rew', 'unrew', 'unrew'] for i in range(len(stay_prob_session))])
    stay_column = np.hstack([np.hstack(stay_prob_session[i]) for i in range(len(stay_prob_session))])
    df_stay_prob = pd.DataFrame(data={'stay': list(stay_column),
                                      'transition': list(transition_column),
                                      'outcome': list(outcome_column),
                                      'subjects': list(subjects_column)})

    r_stay_prob = R.convert_df_to_R(df_stay_prob)
    model = R.afex.aov_car(R.Formula('stay ~ transition * outcome + Error(subjects/ transition * outcome)'),
                           data=r_stay_prob)
    print(R.base.summary(model))
    pairs = R.emmeans.emmeans(model, R.Formula('~transition*outcome'), contr='pairwise', adjust='holm')
    print(pairs)

    with R.localconverter(R.robjects.default_converter + R.pandas2ri.converter):
      p_val = R.robjects.conversion.rpy2py(R.base.summary(pairs).rx2('contrasts')['p.value'])
    pos = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    if plot_significance:
      offset = 0
      for i in range(len(pos)):
        barplot_annotate_brackets(pos[i][0], pos[i][1], p_val[i], np.arange(1, 5), stay_probability_mean, dh=offset)
        offset += 0.2

  ax.set_xlim(0.75, 5)
  if stay_same_diff:
    ax.set_xticks([0, 1, 2, 3, 4], ['\nSame/diff\n\nOutcome', '\nS\n\n1', '\nD\n\n1', '\nS\n\n0', '\nD\n\n0'], fontsize=fontsize)
  else:
    ax.set_xticks([0, 1, 2, 3, 4], ['\nTransition\n\nOutcome', '\nC\n\n1', '\nR\n\n1', '\nC\n\n0', '\nR\n\n0'], fontsize=fontsize)
  ax.set_ylabel('Stay Probability', fontsize=fontsize)
  ax.tick_params(axis='y', labelsize=fontsize)
  ax.tick_params(axis='x', labelsize=fontsize)

def _mixed_model_regression_stay(sessions, predictors=['pr_correct', 'pr_choice', 'pr_outcome', 'pr_trans_CR'],
                                     formula='cbind(stay, switch) ~ pr_correct + pr_choice + pr_trans_CR*pr_outcome + '
                                 '(pr_correct + pr_choice+ pr_trans_CR*pr_outcome | subject_str)', expand_re=False,
                                selection_type='all', select_n='all', block_type='all', opto=False, regressor_type='int',
                                 subjects_virus=[], return_R=False):
  '''
  Predict stay probability based on behavioural variables using mixed effects regression
  '''
  df_sessions = ri.import_regressors(sessions, base_predictors=predictors, lags={}, single_lag=1, selection_type=selection_type,
                                     select_n=select_n, block_type=block_type, stay_pr=True, opto=opto, regressor_type=regressor_type,
                                     subjects_virus=subjects_virus)
  predictors_s = ['subject_str'] + predictors
  if opto == False:
    df_sessions = df_sessions[~df_sessions['forced_choice_next_trial'] & df_sessions['positions_mask']]
  else:
    df_sessions = df_sessions[~df_sessions['forced_choice_next_trial'] & df_sessions['positions_mask']
                                       & df_sessions['same_prob_stim']]
  print('# Trials: {}'.format(len(df_sessions)))
  df_sessions = df_sessions \
    .groupby(predictors_s)['stay'] \
    .agg({('stay', lambda stay: sum(stay)),
          ('switch', lambda stay: sum(~stay))}).reset_index()

  R_sessions = R.convert_df_to_R(df_sessions)
  nc = R.parallel.detectCores()  # number of cores
  cl = R.parallel.makeCluster(R.robjects.r.rep('localhost', nc))  # make cluster

  model = R.afex.mixed(R.Formula(formula),
                       data=R_sessions,
                       family='binomial',
                       cl=cl,
                       method='LRT',
                       all_fit=True,
                       expand_re=expand_re)
  print(R.base.summary(model))
  print(R.afex.nice(model))
  coef = R.base.summary(model).rx2('coefficients')

  with R.localconverter(R.robjects.default_converter + R.pandas2ri.converter):
    coef_py = R.robjects.conversion.rpy2py(coef)
  df_mixed = pd.DataFrame(coef_py, index=coef.names[0], columns=coef.names[1])

  if return_R:
    return df_mixed, model
  else:
    return df_mixed, []

def plot_mixed_model_regression_stay(sessions, predictors=['pr_correct', 'pr_choice', 'pr_outcome', 'pr_trans_CR'],
                                     formula='cbind(stay, switch) ~ pr_correct + pr_choice + pr_trans_CR*pr_outcome + '
                                 '(pr_correct + pr_choice+ pr_trans_CR*pr_outcome | subject)', expand_re=False,
                                selection_type='all', select_n='all', block_type='all', pltfigure=(2.5, 3.3), title=[],
                                     fontsize=9, ticks_formula_names=[], opto=False, plot_separately=[], ylim=[], save=[],
                                     regressor_type='int', return_results=False, subjects_virus=[], return_R=False, df_mixed_input=[]):
  '''
  Plot mixed effect regression model predicting stay
  '''
  if type(df_mixed_input) != pd.core.frame.DataFrame:
    df_mixed, model = _mixed_model_regression_stay(sessions, predictors=predictors, formula=formula, expand_re=expand_re,
                                            selection_type=selection_type, select_n=select_n, block_type=block_type,
                                            opto=opto, regressor_type=regressor_type, subjects_virus=subjects_virus, return_R=return_R)
  else:
    df_mixed = df_mixed_input

  if not plot_separately:
    plot_separately = [list(np.arange(len(df_mixed.index.values)))]

  for i,x in enumerate(plot_separately):
    x_ticks = df_mixed.index.values[x]
    estimates = df_mixed['Estimate'].values[x]
    std = df_mixed['Std. Error'].values[x]
    p_val = df_mixed['Pr(>|z|)'].values[x]
    x_labels = x_ticks

    if ticks_formula_names != []:
      pos, x_ticks = zip(*[(i, xt) for i, xt in enumerate(x_ticks) if xt in ticks_formula_names[0]])
      estimates = estimates[list(pos)]
      std = std[list(pos)]
      p_val = p_val[list(pos)]
      pos_x_label = [ticks_formula_names[0].index(xt) for xt in x_ticks]
      x_labels = np.asarray(ticks_formula_names[1])[pos_x_label]

    if pltfigure != False:
      plt.figure(figsize=pltfigure)

    plt.errorbar(range(len(x_labels)), estimates, std, linestyle='none', elinewidth=3, capsize=5, color='k')

    [plt.text(x, y, text, ha='center') for x, y, text in zip(range(len(x_labels)), estimates + std + 0.005, stats_annotation(p_val))]

    plt.ylabel('Regression Coefficient', fontsize=fontsize)
    plt.axhline(0, linestyle='--', color='k', linewidth=0.5)

    if title != []:
        plt.title(title)

    if ylim:
      if ylim[i]:
        plt.ylim(ylim[i])

    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, fontsize=fontsize, rotation=45)
    plt.yticks(fontsize=fontsize)
    plt.gcf().set_tight_layout(True)

    if save:
      if save[i]:
        savefig(save[i][0], save[i][1])

  if return_results and type(df_mixed_input) != pd.core.frame.DataFrame:
    if return_R:
      return df_mixed, model
    else:
      return df_mixed

def _compute_logistic_regression(sessions, base_predictors, block_type, selection_type, select_n, lags, single_lag, log,
                                 ave_predictors, lag_fixed_predictors, sum_predictors, lags_future=False,
                                 same_prob_stim=False, concatenate_sessions=False, regres_solver='newton-cg'):
  '''
  Compute fixed effects logistic regression model
  selection_type: 'all', 'before', 'after', 'xtr'
  concatenate_sessions: True to run the regression for all sessions together; False to run regression for each session separately
  '''

  coef_reg = []
  all_predictors = []
  all_choices = []

  if type(ave_predictors) is list:
    ave_predictors = np.concatenate(([0], ave_predictors))
    ave_predictors_add = ave_predictors
    for i in range(len(base_predictors) - 1):
      ave_predictors_add = ave_predictors_add + lags
      ave_predictors = np.concatenate((ave_predictors, ave_predictors_add))

  if type(sum_predictors) is list:
    sum_predictors = [[x - 1 for x in sum_predictors[i]] for i in range(len(sum_predictors))]
    sum_predictors1 = sum_predictors[:]
    lags_base = [lags * i for i in range(1, len(base_predictors))]
    for lb in lags_base:
      sum_predictors1 += [[x + lb for x in sum_predictors[i]] for i in range(len(sum_predictors))]

  for session in sessions:
    choices, transitions_AB, second_steps, outcomes, trans_state, transitions_CR, \
    transition_CR_x_outcome, forced_choice_trials, rew_state = _get_data_to_analyse(session)

    data_to_analyse = choices, transitions_AB, second_steps, outcomes, trans_state, transitions_CR, \
                      transition_CR_x_outcome, forced_choice_trials, rew_state

    if lags_future:
      choices, predictors_past, predictors_future = _get_predictors(
          data_to_analyse, base_predictors, lags, single_lag=single_lag, lags_future=True)
      if lags_future in ['stim_future', 'non_stim_future']:
        predictors = predictors_future[:]
      else:
        predictors = np.hstack((predictors_past, predictors_future))
    else:
      choices, predictors = _get_predictors(data_to_analyse, base_predictors, lags, single_lag=single_lag, session=session)

    # select trials to analyse
    if lags_future == 'stim_future':
      positions = session.trial_data['stim'].astype(bool)
    elif lags_future == 'non_stim_future':
      positions = ~session.trial_data['stim'].astype(bool)
    else:
      positions = session.select_trials(selection_type=selection_type, select_n=select_n, block_type=block_type)
    choices = choices[positions]
    predictors = predictors[positions]
    forced_choice_trials = forced_choice_trials[positions]

    # eliminate forced choice trials
    if same_prob_stim == False:
      choices = [choices[i] for i in range(len(choices)) if forced_choice_trials[i] == False]
      predictors = [predictors[i] for i in range(len(predictors)) if forced_choice_trials[i] == False]
    else:
      choices = [choices[i] for i in range(len(choices)) if (forced_choice_trials[i] == False) and
                 (i - 1 not in np.where(session.trial_data['stim'][0])) and
                 (i - 2 not in np.where(session.trial_data['stim'][0])) and
                 (i - 3 not in np.where(session.trial_data['stim'][0]))]
      predictors = [predictors[i] for i in range(len(predictors)) if (forced_choice_trials[i] == False) and
                    (i - 1 not in np.where(session.trial_data['stim'][0])) and
                    (i - 2 not in np.where(session.trial_data['stim'][0])) and
                    (i - 3 not in np.where(session.trial_data['stim'][0]))]


    if type(ave_predictors) is np.ndarray:
      predictors = [np.asarray([np.nanmean(predictors[s][ave_predictors[i]:ave_predictors[i + 1]]) \
                                for i in range(len(ave_predictors) - 1) if ave_predictors[i] < ave_predictors[i + 1]]) \
                    for s in range(len(predictors))]

    if type(sum_predictors) is list:
      predictors = [[np.sum([predictors[t][x] for x in sum_predictors1[i]]) for i in range(len(sum_predictors1))]
                    for t in range(len(predictors))]

    if lag_fixed_predictors != []:
      _, predictors_fix = _get_predictors(data_to_analyse, lag_fixed_predictors, {}, single_lag=single_lag)
      predictors_fix = [predictors_fix[i] for i in range(len(predictors_fix)) if forced_choice_trials[i] == False]
      predictors = np.hstack((predictors, predictors_fix))

    if concatenate_sessions == False:

      if (predictors != []) and (len(set(choices)) != 1): # ValueError: This solver needs samples of at least 2 classes
        # in the data, but the data contains only one class: 0. This is because of a bug in sklearn.linear_model module.
        # Does not work if there is just one label, e.g. on the trials to analyse, the animal always made the same choice

        if log == True:
          log_reg = lm.LogisticRegression(penalty='none', solver=regres_solver)
        else:
          log_reg = lm.LinearRegression()

        log_reg.fit(predictors, choices)

        coef_reg.append(log_reg.coef_)
    else:
      all_predictors += predictors
      all_choices += choices

  if concatenate_sessions == True:
    if log == True:
      log_reg = lm.LogisticRegression(penalty='none', solver='newton-cg')
    elif log == 'l1':
      log_reg = lm.LogisticRegression(penalty='l1', solver='liblinear')
    else:
      log_reg = lm.LinearRegression()

    log_reg.fit(all_predictors, all_choices)

    coef_reg=log_reg.coef_

  return coef_reg

def plot_log_reg_lagged(sessions, selection_type, block_type, select_n, base_predictors, lags, log=True,
                        ave_predictors=False, lag_fixed_predictors=[], sum_predictors=False, lags_future=False,
                        pltfigure=True, title=[], legend=True, legend_names=[], colors='k', scatter=False, fontsize=8,
                        same_prob_stim=False, color_per_subject=[], return_scatter=False, regres_solver='newton-cg', ttest=True,
                        ax=None):
  '''
  Plot lagged fixed effects logistic regression model predicting stay
  '''
  trial_lags = lags
  single_lag = {}
  sessions_sub = []
  subjects = list(set([sessions[i].subject_ID for i in range(len(sessions))]))
  for sub in subjects:
    idx_sub = np.where([sessions[i].subject_ID == sub for i in range(len(sessions))])[0]
    sessions_sub.append([sessions[x] for x in idx_sub])

  coef_reg_all = pp.map(partial(_compute_logistic_regression, base_predictors=base_predictors, block_type=block_type,
                            selection_type=selection_type, select_n=select_n, lags=lags, single_lag=single_lag,
                            log=log, ave_predictors=ave_predictors, lag_fixed_predictors=lag_fixed_predictors,
                            sum_predictors=sum_predictors, lags_future=lags_future, same_prob_stim=same_prob_stim,
                                concatenate_sessions=True, regres_solver=regres_solver), sessions_sub)
  coef_reg_all = [np.vstack(x).T for x in coef_reg_all]

  if lag_fixed_predictors == []:
    coef_reg_sub = [np.mean(coef, axis=1) for coef in coef_reg_all]
    sem_sub = [stats.sem(coef, axis=1) for coef in coef_reg_all]
    mean = np.mean(coef_reg_sub, axis=0)
    if len(coef_reg_all) == 1:
      sem = stats.sem(coef_reg_all[0], axis=1)
    else:
      sem = stats.sem(coef_reg_sub, axis=0)
  else:
    #  maybe it doesn't work
    coef_reg_sub = np.mean([coef[:-lag_fixed_predictors] for coef in coef_reg_all], axis=2)
    mean = np.mean(coef_reg_sub, axis=0)
    sem = stats.sem(coef_reg_sub, axis=0)

  # Split by predictor
  if lags_future == True:
    mean_past = mean[:len(base_predictors) * lags]
    sem_past = sem[:len(base_predictors) * lags]
    mean_lag_predictor_past = np.split(mean_past, len(base_predictors))
    mean_lag_predictor_past = np.asarray([m[::-1] for m in np.asarray(mean_lag_predictor_past)])
    sem_lag_predictor_past = np.split(sem_past, len(base_predictors))
    sem_lag_predictor_past = np.asarray([m[::-1] for m in np.asarray(sem_lag_predictor_past)])

    mean_future = mean[len(base_predictors) * lags:]
    sem_future = sem[len(base_predictors) * lags:]
    mean_lag_predictor_future = np.asarray(np.split(mean_future, len(base_predictors)))
    sem_lag_predictor_future = np.asarray(np.split(sem_future, len(base_predictors)))

    mean_lag_predictor = np.hstack((mean_lag_predictor_past, mean_lag_predictor_future))
    sem_lag_predictor = np.hstack((sem_lag_predictor_past, sem_lag_predictor_future))

  else:
    mean_lag_predictor = np.split(mean, len(base_predictors))
    sem_lag_predictor = np.split(sem, len(base_predictors))

  if pltfigure != False:
    ax = ax or plt.gca(figsize=(10, 8))
  else:
    ax = ax or plt.gca()
  if type(ave_predictors) is list:
    x = range(len(ave_predictors))
  elif type(sum_predictors) is list:
    x = range(len(sum_predictors))
  elif lags_future:
    x = np.concatenate((np.arange(-lags, 0), np.arange(1, lags+1)))
  else:
    x = range(trial_lags)
  if legend_names == []:
    legend_names = base_predictors
  if type(colors) == str:
    colors = [colors] * len(base_predictors)
  if ttest:
    t, prob = stats.ttest_1samp([np.asarray(coef_reg_sub)[i] for i in range(len(np.asarray(coef_reg_sub)))], 0, axis=0)
    p_val = statsmodels.stats.multitest.multipletests(prob, method='bonferroni')[1]
    p_val_lag_predictor = np.split(p_val, len(base_predictors))
    max_y = np.max(mean_lag_predictor) + np.max(sem_lag_predictor)
  add_y = 0.02
  for i in range(len(base_predictors)):
    ax.errorbar(x, mean_lag_predictor[i], yerr=sem_lag_predictor[i], label=legend_names[i], color=colors[i], capsize=5)
    if ttest:
      [ax.text(x, y, text, ha='center', color=colors[i], fontsize=8) for x, y, text in
       zip(range(len(p_val_lag_predictor[i])), np.repeat(max_y + add_y, len(p_val_lag_predictor[i])), stats_annotation(p_val_lag_predictor[i]))]
      add_y += 0.02

  if legend ==True:
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=5, prop={'size': 8})

  if scatter == True or scatter =='errorbar':
    x = np.random.normal(0, 0.14, size=len(coef_reg_sub))
    x_return = np.repeat(0, len(coef_reg_sub))
    y = np.vstack(coef_reg_sub)
    if scatter == 'errorbar':
      sy = np.vstack(sem_sub)
    x_repeat = np.asarray([x] * len(sum_predictors) * len(base_predictors)).T
    x_repeat_return = np.asarray([x_return] * len(sum_predictors) * len(base_predictors)).T
    sum_pred_repeat = np.vstack([np.hstack([np.arange(len(sum_predictors))] * len(base_predictors))] * len(y))
    x_scatter = x_repeat + sum_pred_repeat
    x_scatter_return = x_repeat_return + sum_pred_repeat
    colors_scatter = np.vstack([np.repeat(colors, len(sum_predictors))] * len(y))
    if color_per_subject: # this is only used for per subject errorbar
      colors = [color_per_subject[s] for s in subjects]
      markers = [ax.Line2D([0], [0], marker='o', color=c, ls="", mec=None, markersize=10) for c in colors] #not sure this will work with ax
      labels = subjects
      ax.legend(markers, labels)
    else:
      colors = 'grey'
    for i in range(len(y.T)):
      if scatter == True:
        ax.scatter(x_scatter.T[i], y.T[i], c=colors_scatter.T[i], s=5, alpha=0.7, lw=0)
      else:
        for si, s in enumerate(subjects):
          ax.errorbar(x_scatter.T[i][si], y.T[i][si], sy.T[i][si], c=colors[si], linestyle='none', elinewidth=1, capsize=3)

  if not lags_future:
    ax.invert_xaxis()
  if ave_predictors:
    ax.set_xticks(range(len(ave_predictors)))
    ax.set_xticklabels(list(np.negative(ave_predictors)))
  elif sum_predictors and not lags_future:
    ax.set_xticks(range(len(sum_predictors)))
    ax.set_xticklabels(list([[-x for x in sum_predictors[i]] for i in range(len(sum_predictors))]))
  elif sum_predictors and lags_future:
    ax.set_xticks((range(len(sum_predictors))))
    ax.set_xticklabels(list([[x for x in sum_predictors[i]] for i in range(len(sum_predictors))]))
  elif lags_future:
    ax.axvline(x=0, color='k')
  else:
    ax.set_xticks(range(lags))
    ax.set_xticklabels(list(reversed(range(-lags, 0))), fontsize=fontsize)
  ax.axhline(y=0, color='k', ls='--')
  ax.set_ylabel('Regression Coefficient', fontsize=fontsize)
  ax.set_xlabel('Lag (Trials)', fontsize=fontsize)
  if log == True:
    regression = 'Log reg'
  else:
    regression = 'Linear reg'
  if title != []:
    ax.set_title(title)
  else:
    ax.set_title('{} trials - {}'.format(block_type, regression))

  if pltfigure != False:
    plt.set_tight_layout(True)

  if return_scatter == True:
    return x_scatter_return, y, sum_pred_repeat, coef_reg_all

def _get_data_to_analyse(session, transform_rew_state=True):
  '''
  unpack data into single variables
  '''
  if hasattr(session, 'unpack_trial_data'):
    choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data(dtype=bool)
  else:
    choices = session.trial_data['choices'].astype(bool)
    transitions_AB = session.trial_data['transitions'].astype(bool)
    second_steps = session.trial_data['second_steps'].astype(bool)
    outcomes = session.trial_data['outcomes'].astype(bool)
  trans_state = session.blocks['trial_trans_state']  # Trial by trial state of the tranistion matrix (A vs B)
  transitions_CR = transitions_AB == trans_state  # Trial by trial common or rare transitions (True: common; False: rare)
  transition_CR_x_outcome = transitions_CR == outcomes  # True: common and rewarded / rare and unrewarded
  # False: common and unrewarded / rare and rewarded
  forced_choice_trials = ~session.trial_data['free_choice']


  # Transform blocks['trial_rew_state'] to rew_state so up == 2, neutral == 1, down == 0
  if transform_rew_state == True:
    rew_state = np.array([session.blocks['trial_rew_state'][i] + 1 if session.blocks['trial_rew_state'][i] == 1
                          else session.blocks['trial_rew_state'][i] - 1 if session.blocks['trial_rew_state'][i] == 2
    else session.blocks['trial_rew_state'][i] for i in range(len(session.blocks['trial_rew_state']))])
  else:
    rew_state = session.blocks['trial_rew_state']

  return (choices, transitions_AB, second_steps, outcomes, trans_state, transitions_CR, transition_CR_x_outcome,
          forced_choice_trials, rew_state)

def _get_predictors(data_to_analyse, base_predictors, lags, single_lag, session=[], lags_future=False, fits=[],
                    fits_names=[], hemisphere=[], Q_values=[], extra_predictors=[], timed_data_names=[]):
  '''
  return predictors value per trial defined in 'base_predictors'
  '''

  choices, transitions_AB, second_steps, outcomes, trans_state, transitions_CR, transition_CR_x_outcome, \
  forced_choice_trials, rew_state = data_to_analyse

  base_predictors = [bp for bp in base_predictors if bp not in timed_data_names]
  if type(lags) == int:
    lags = {p: lags for p in base_predictors}

  predictors_lags = []  # predictor names including lags.
  for predictor in base_predictors:
    if predictor in list(lags.keys()):
      for i in range(lags[predictor]):
        predictors_lags.append(predictor + '-' + str(i + 1))  # Lag is indicated by value after '-' in name.
    else:
      predictors_lags.append(predictor)  # If no lag specified, defaults to 0.

  n_predictors = len(predictors_lags)

  correct = 0.5 * (rew_state - 1) * (
          2 * trans_state - 1)  # 0.5, 0, -0.5 for left poke being correct, neutral, incorrect option.

  choices_l1 = _lag(choices, 1)
  choices_l2 = _lag(choices, 2)
  choices_f1 = _lag(choices, -1)
  second_steps_l1 = _lag(second_steps, 1)
  reward_l1 = _lag(outcomes, 1)
  reward_f1 = _lag(outcomes, -1)
  trans_state_l1 = _lag(trans_state, 1)
  rew_state_l1 = _lag(rew_state, 1)

  same_ch = choices == choices_l1
  choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
  causal_mb = choices_to_ssl1 == choices
  same_ss = second_steps == second_steps_l1

  reward_x = [x * 1 if x == True else -1 for x in outcomes]
  reward_l1_x = [x * 1 if x == True else -1 for x in reward_l1]

  bp_values = {}

  for p in base_predictors:

    if p == 'correct':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
      bp_values[p] = correct

    if p == 'correct_stim':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
      bp_values[p] = [x if session.trial_data['stim'][i] == 1 else 0 for i, x in enumerate(correct)]
    if p == 'correct_nonstim':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
      bp_values[p] = [x if session.trial_data['stim'][i] == 0 else 0 for i, x in enumerate(correct)]

    elif p == 'side':  # 0.5, -0.5 for up, down poke reached at second step.
      bp_values[p] = second_steps - 0.5

    elif p == 'choice':  # 0.5, - 0.5 for choices left, right.
      bp_values[p] = choices - 0.5
    elif p == 'choice_stim':  # 0.5, - 0.5 for choices left, right.
      stim = session.trial_data['stim']
      bp_values[p] = (choices & stim) - 0.5
    elif p == 'choice_nonstim':  # 0.5, - 0.5 for choices left, right.
      stim = session.trial_data['stim']
      bp_values[p] = (choices & ~stim) - 0.5

    elif p == 'outcome':  # 0.5 , -0.5 for  rewarded , not rewarded.
      bp_values[p] = (outcomes == choices) - 0.5
    elif p == 'outcome_stim':  # 0.5 , -0.5 for  rewarded , not rewarded.
      bp_values[p] = (outcomes & stim) * (choices - 0.5)
    elif p == 'outcome_nonstim':  # 0.5 , -0.5 for  rewarded , not rewarded.
      bp_values[p] = (outcomes & ~stim) * (choices - 0.5)

    elif p == 'stim':
      bp_values[p] = session.trial_data['stim'] - 0.5

    elif p == 'choice_x_stim':
      stim = session.trial_data['stim']
      bp_values[p] = (choices == stim) -0.5

    elif p == 'trans_x_stim':
      stim = session.trial_data['stim']
      bp_values[p] = ((transitions_CR == stim) == choices) - 0.5

    elif p == 'out_x_stim':
      stim = session.trial_data['stim']
      bp_values[p] = ((outcomes == stim) == choices) - 0.5

    elif p == 'trCR_x_out_x_stim':
      stim = session.trial_data['stim']
      bp_values[p] = ((transition_CR_x_outcome == stim) == choices) - 0.5

    elif p == 'forced_choice':
      bp_values[p] = (forced_choice_trials == choices) - 0.5

    elif p == 'stay': # 0.5 stay, -0.5 switch
      bp_values[p] = list(np.hstack(([0], (choices[1:] == choices[:-1]).astype(int) - 0.5)))

    elif p == 'same_ch_same_ss':
      bp_values[p] = ((choices == choices_l1) & (second_steps == second_steps_l1)) * (choices - 0.5)
      bp_values[p][0] = 0

    elif p == 'same_ch_diff_ss':
      bp_values[p] = ((choices == choices_l1) & (second_steps != second_steps_l1)) * (choices - 0.5)
      bp_values[p][0] = 0

    elif p == 'diff_ch_same_ss':
      bp_values[p] = ((choices != choices_l1) & (second_steps == second_steps_l1)) * (choices - 0.5)
      bp_values[p][0] = 0

    elif p == 'diff_ch_diff_ss':
      bp_values[p] = ((choices != choices_l1) & (second_steps != second_steps_l1)) * (choices - 0.5)
      bp_values[p][0] = 0

    elif p == 'out_x_same_ch_same_ss':
      bp_values[p] = ((choices == choices_l1) & (second_steps == second_steps_l1)) * (
              (outcomes == choices) - 0.5)
      bp_values[p][0] = 0

    elif p == 'out_x_same_ch_diff_ss':
      bp_values[p] = ((choices == choices_l1) & (second_steps != second_steps_l1)) * (
              (outcomes == choices) - 0.5)
      bp_values[p][0] = 0

    elif p == 'out_x_diff_ch_same_ss':
      bp_values[p] = ((choices != choices_l1) & (second_steps == second_steps_l1)) * (
              (outcomes == choices) - 0.5)
      bp_values[p][0] = 0

    elif p == 'out_x_diff_ch_diff_ss':
      bp_values[p] = ((choices != choices_l1) & (second_steps != second_steps_l1)) * (
              (outcomes == choices) - 0.5)
      bp_values[p][0] = 0

    elif p == 'same_ch':
      bp_values[p] = ((choices == choices_l1) == choices) - 0.5
      bp_values[p][0] = 0

    elif p == 'same_ss':
      bp_values[p] = ((second_steps == second_steps_l1) == choices) - 0.5
      bp_values[p][0] = 0

    elif p == 'same_ch_x_same_ss':
      bp_values[p] = (((choices == choices_l1) == (second_steps == second_steps_l1))
                      == choices) - 0.5
      bp_values[p][0] = 0

    elif p == 'out_x_same_ch':
      bp_values[p] = ((choices == choices_l1) == (outcomes == choices)) - 0.5
      bp_values[p][0] = 0

    elif p == 'out_x_same_ss':
      bp_values[p] = ((second_steps == second_steps_l1) == (outcomes == choices)) - 0.5
      bp_values[p][0] = 0

    elif p == 'out_x_same_ch_x_same_ss':
      bp_values[p] = (((choices == choices_l1) == (second_steps == second_steps_l1))
                      == (outcomes == choices)) - 0.5
      bp_values[p][0] = 0

    elif p == 'same_ss_if_same_ch':
      bp_values[p] = (((second_steps == second_steps_l1) == choices) - 0.5) * (choices == choices_l1)
      bp_values[p][0] = 0

    elif p == 'same_ss_not_same_ch':
      bp_values[p] = (((second_steps == second_steps_l1) == choices) - 0.5) * (choices != choices_l1)
      bp_values[p][0] = 0

    elif p == 'trans_CR':  # 0.5, -0.5 for common, rare transitions.
      bp_values[p] = ((transitions_CR) == choices) - 0.5
    elif p == 'trans_CR_stim':  # 0.5, -0.5 for common, rare transitions.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (transitions_CR & stim) * (choices - 0.5)
    elif p == 'trans_CR_nonstim':  # 0.5, -0.5 for common, rare transitions.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (transitions_CR & ~stim) * (choices - 0.5)

    elif p == 'trCR_x_out':  # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
      bp_values[p] = (transition_CR_x_outcome == choices) - 0.5
    elif p == 'trCR_x_out_stim':  # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (transition_CR_x_outcome & stim) * (choices - 0.5)
    elif p == 'trCR_x_out_nonstim':  # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (transition_CR_x_outcome & ~stim) * (choices - 0.5)

    elif p == 'rew_com':  # Rewarded common transition predicts repeating choice.
      rew_com = (outcomes & transitions_CR) * (choices - 0.5)
      bp_values[p] = np.asarray([1 if rc==0.5 else -1 if rc==-0.5 else 0 for rc in rew_com])

    elif p == 'rew_rare':  # Rewarded rare transition predicts repeating choice.
      rew_rare = (outcomes & ~transitions_CR) * (choices - 0.5)
      bp_values[p] = np.asarray([1 if rr==0.5 else -1 if rr==-0.5 else 0 for rr in rew_rare])

    elif p == 'non_com':  # Non-rewarded common transition predicts repeating choice.
      non_com = (~outcomes & transitions_CR) * (choices - 0.5)
      bp_values[p] = np.asarray([1 if rc==0.5 else -1 if rc==-0.5 else 0 for rc in non_com])

    elif p == 'non_rare':  # Non-Rewarded rare transition predicts repeating choice.
      non_rare = (~outcomes & ~transitions_CR) * (choices - 0.5)
      bp_values[p] = np.asarray([1 if nr==0.5 else -1 if nr==-0.5 else 0 for nr in non_rare])

    #Merima's regressors
    elif p == 'tr_rew':  # common rewarded and rare rewarded predicts repeating choice
      ComRewTrials = outcomes & transitions_CR
      RareRewTrials = -1 * (outcomes & ~transitions_CR)
      bp_values[p] = (ComRewTrials + RareRewTrials) * (choices - 0.5)
    elif p == 'tr_nonrew':  # common unrewarded and rare unrewarded predicts repeating choice
      ComNRewTrials = ~outcomes & transitions_CR
      RareNRewTrials = -1 * (~outcomes & ~transitions_CR)
      bp_values[p] = (ComNRewTrials + RareNRewTrials) * (choices - 0.5)

    elif p == 'rew_com_stim':  # Rewarded common transition stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (outcomes & transitions_CR & stim) * (choices - 0.5)

    elif p == 'rew_rare_stim':  # Rewarded rare transition stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (outcomes & ~transitions_CR & stim) * (choices - 0.5)

    elif p == 'non_com_stim':  # Non-rewarded common transition stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~outcomes & transitions_CR & stim) * (choices - 0.5)

    elif p == 'non_rare_stim':  # Non-Rewarded rare transition stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~outcomes & ~transitions_CR & stim) * (choices - 0.5)

    elif p == 'rew_com_nonstim':  # Rewarded common transition non-stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (outcomes & transitions_CR & ~stim) * (choices - 0.5)

    elif p == 'rew_rare_nonstim':  # Rewarded rare transition non-stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (outcomes & ~transitions_CR & ~stim) * (choices - 0.5)

    elif p == 'non_com_nonstim':  # Non-rewarded common transition non-stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~outcomes & transitions_CR & ~stim) * (choices - 0.5)

    elif p == 'non_rare_nonstim':  # Non-Rewarded rare transition non-stimulated predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~outcomes & ~transitions_CR & ~stim) * (choices - 0.5)

    elif p == 'stim_com': # Stim common transitions predicts repeating choice
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (stim & transitions_CR) * (choices - 0.5)

    elif p == 'stim_rare':  # Stim rare transition predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (stim & ~transitions_CR) * (choices - 0.5)

    elif p == 'nonstim_com':  # Non-stim common transition predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~stim & transitions_CR) * (choices - 0.5)

    elif p == 'nonstim_rare':  # Non-stim rare transition predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~stim & ~transitions_CR) * (choices - 0.5)

    elif p == 'rew_stim':  # Rewarded stimulated trials predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (outcomes & stim) * (choices - 0.5)

    elif p == 'rew_nonstim':  # Rewarded non-stim trials predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (outcomes & ~stim) * (choices - 0.5)

    elif p == 'non_stim':  # Non-rewarded stim predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~outcomes & stim) * (choices - 0.5)

    elif p == 'non_nonstim':  # Non-Rewarded non-stim trials predicts repeating choice.
      stim = session.trial_data['stim'].astype(bool)
      bp_values[p] = (~outcomes & ~stim) * (choices - 0.5)

    elif p == 'reward':  # 1 rewarded trials, 0 non-rewarded trials
      bp_values[p] = np.asarray([0.5 if o == 1 else -0.5 for o in (outcomes * 1)])
    elif p == 'reward_str':  # 1 rewarded trials, 0 non-rewarded trials
      bp_values[p] = np.asarray(['rew' if o == 1 else 'nonrew' for o in (outcomes * 1)])
    elif p == 'ch':  # 1 rewarded trials, 0 non-rewarded trials
      bp_values[p] = choices * 1
    elif p == 'ch_1':  # 1 rewarded trials, 0 non-rewarded trials
      bp_values[p] = choices_l1 * 1

    elif p == 'good_ss':  # 0.5 good second-step, -0.5 bad second_step, 0 neutral block
      bp_values[p] = np.asarray([0.5 if (ss == rs and rs != 2) else -0.5 if (ss != rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
    elif p == 'good_ss_1':  # 0.5 good second-step, -0.5 bad second_step, 0 neutral block, previous trial
      bp_values[p] = _lag(np.asarray([0.5 if (ss == rs and rs != 2) else -0.5 if (ss != rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)]), 1)
    elif p == 'good_ss_single':  # 1 good second-step, 0 bad second_step, 0 neutral block
      bp_values[p] = np.asarray([1 if (ss == rs and rs != 2) else 0 if (ss != rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
    elif p == 'bad_ss_single':  # 0 good second-step, 1 bad second_step, 0 neutral block
      bp_values[p] = np.asarray([0 if (ss == rs and rs != 2) else 1 if (ss != rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
    elif p == 'neutral_ss_single':  # 0 good second-step, 0 bad second_step, 1 neutral block
      bp_values[p] = np.asarray([0 if (ss == rs and rs != 2) else 0 if (ss != rs and rs != 2) else 1
                                 for ss, rs in zip(second_steps, rew_state)])

    elif p == 'good_bad_ss_x_reward':  # 0.5 bad_second step - non_rewarded /  good_Second step - rewarded else -0.5
                                       # 0 if neutral block
      bp_values[p] = np.asarray([0.5 if (x == True and rs != 2) else -0.5 if (x == False and rs != 2) else 0
                                 for x, rs in zip((outcomes == (second_steps == rew_state)), rew_state)])

    elif p == 'ss_ssl1_x_rewl1':  # 1 previous ss == current ss and reward on l1
      bp_values[p] = ((second_steps == second_steps_l1) == reward_l1) * 1

    elif p == 'reward_1':  # 1 rewarded trials in the previous trial
      bp_values[p] = reward_l1 * 1
    elif p == 'reward_f1':  # 1 rewarded trials in the previous trial
      bp_values[p] = reward_f1 * 1

    elif p == 'reward_2':  # 1 rewarded trials in the previous trial
      bp_values[p] = _lag(outcomes, 2)

    elif p == 'reward_3':  # 1 rewarded trials in the previous trial
      bp_values[p] = _lag(outcomes, 3)
    elif p == 'reward_4':  # 1 rewarded trials in the previous trial
      bp_values[p] = _lag(outcomes, 4)
    elif p == 'reward_5':  # 1 rewarded trials in the previous trial
      bp_values[p] = _lag(outcomes, 5)

    elif p == 'cum_rew': # cumulative reward
      bp_values[p] = list(np.cumsum(outcomes))

    elif p == 'ntrial': # trial number
      bp_values[p] = np.arange(len(choices))

    elif p == 'free_choice_single': # 1 free choice trial, 0 forced choice
      bp_values[p] = ~forced_choice_trials * 1

    elif p == 'forced_choice_single': # 1 forced, 0 free
      bp_values[p] = np.asarray([0.5 if x ==1 else -0.5 for x in forced_choice_trials * 1])

    elif p == 'forced_choice_x_correct': # +0.5 forced correct, -0.5 forced incorrect, 0 free
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = np.asarray([0 if rs == 2 else 0.5 if ((c == True) and (f == True))
                    else -0.5 for c, rs, f in zip(correct, rew_state, forced_choice_trials)])
    elif p == 'free_choice_x_correct': # +0.5 forced correct, -0.5 forced incorrect, 0 free
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = np.asarray([0 if rs == 2 else 0.5 if ((c == True) and (f == False))
                    else -0.5 for c, rs, f in zip(correct, rew_state, forced_choice_trials)])

    elif p == 'common_rare':  # 0.5 common transition, -0.5 rare transition
      bp_values[p] = [0.5 if t == True else -0.5 for t in transitions_CR]

    elif p == 'forced_choice_x_transition':
      common_rare = np.asarray([0.5 if t == True else -0.5 for t in transitions_CR])
      bp_values[p] = common_rare * forced_choice_trials
    elif p == 'free_choice_x_transition':
      common_rare = np.asarray([0.5 if t == True else -0.5 for t in transitions_CR])
      bp_values[p] = common_rare *  ~forced_choice_trials

    elif p == 'forced_choice_x_correct':
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      correct_p = np.asarray([0.5 if c == True else -0.5 for c in correct])
      bp_values[p] = correct_p * forced_choice_trials
    elif p == 'free_choice_x_correct':
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      correct_p = np.asarray([0.5 if c == True else -0.5 for c in correct])
      bp_values[p] = correct_p * ~forced_choice_trials

    elif p == 'common_x_free_forced':
      free_forced = np.asarray([-0.5 if f == True else 0.5 for f in forced_choice_trials])
      bp_values[p] = free_forced * transitions_CR
    elif p == 'rare_x_free_forced':
      free_forced = np.asarray([-0.5 if f == True else 0.5 for f in forced_choice_trials])
      bp_values[p] = free_forced * ~transitions_CR

    elif p == 'common_transition':
      bp_values[p] = transitions_CR * 1
    elif p == 'rare_transition':
      bp_values[p] = ~transitions_CR * 1

    elif p == 'common_good_ss':
      good_ss = np.asarray([0.5 if (ss == rs and rs != 2) else -0.5 if (ss != rs and rs != 2) else 0
                  for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_ss * transitions_CR
    elif p == 'rare_good_ss':
      good_ss = np.asarray([0.5 if (ss == rs and rs != 2) else -0.5 if (ss != rs and rs != 2) else 0
                  for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_ss * ~transitions_CR

    elif p == 'common_rare_x_rew':  # 0.5 common-rew / rare-unrew
      bp_values[p] = [0.5 if t == True else -0.5 for t in transition_CR_x_outcome]

    elif p == 'common_rare_1_x_rew_1':  # 1 common-rew / rare-unrew
      transition_CR_x_outcome_1 = _lag(transition_CR_x_outcome, 1)
      bp_values[p] = transition_CR_x_outcome_1 * 1

    elif p == 'common_x_rew':  # +0.5 common-rew / -0.5 common-non_rew / 0 rare transitions
      rew = [0.5 if r == True else -0.5 for r in outcomes]
      bp_values[p] = transition_CR_x_outcome * rew

    elif p == 'rare_x_rew':  # +0.5 rare-rew / -0.5 rare-non_rew / 0 common transitions
      rew = [0.5 if r ==  True else -0.5 for r in outcomes]
      bp_values[p] = ~transition_CR_x_outcome * rew
    elif p == 'common_x_nonrew':  # +0.5 common-rew / -0.5 common-non_rew / 0 rare transitions
      rew = [0.5 if r == False else -0.5 for r in outcomes]
      bp_values[p] = transition_CR_x_outcome * rew

    elif p == 'rare_x_nonrew':  # +0.5 rare-rew / -0.5 rare-non_rew / 0 common transitions
      rew = [0.5 if r ==  False else -0.5 for r in outcomes]
      bp_values[p] = ~transition_CR_x_outcome * rew

    elif p == 'common_1_x_rew_1':  # +0.5 common-rew / -0.5 common-non_rew / 0 rare transitions
      rew = np.asarray([0.5 if r == True else -0.5 for r in outcomes])
      transition_CR_x_outcome_1 = _lag(transition_CR_x_outcome, 1)
      rew_1 = _lag(rew, 1)
      bp_values[p] = transition_CR_x_outcome_1 * rew_1

    elif p == 'rare_1_x_rew_1':  # +0.5 rare-rew / -0.5 rare-non_rew / 0 common transitions
      rew = np.asarray([0.5 if r == True else -0.5 for r in outcomes])
      transition_CR_x_outcome_1 = _lag(transition_CR_x_outcome, 1)
      rew_1 = _lag(rew, 1)
      bp_values[p] = ~transition_CR_x_outcome_1 * rew_1

    elif p == 'common_x_rew_1':  # +0.5 common-rew1 / -0.5 common-non_rew1 / 0 rare transitions
      rew = np.asarray([0.5 if r == True else -0.5 for r in outcomes])
      rew_1 = _lag(rew, 1)
      bp_values[p] = transition_CR_x_outcome * rew_1

    elif p == 'rare_x_rew_1':  # +0.5 rare-rew1 / -0.5 rare-non_rew1 / 0 common transitions
      rew = np.asarray([0.5 if r == True else -0.5 for r in outcomes])
      rew_1 = _lag(rew, 1)
      bp_values[p] = ~transition_CR_x_outcome * rew_1

    elif p == 'cr_f':  #unitary predictor - common-rew-free
      free_choice = ~forced_choice_trials
      bp_values[p] = (transitions_CR & outcomes & free_choice) * 1
    elif p == 'cnr_f':  #unitary predictor - common-nonrew-free
      free_choice = ~forced_choice_trials
      bp_values[p] = (transitions_CR & ~outcomes & free_choice) * 1
    elif p == 'cr_nf':  #unitary predictor - common-rew-forced
      free_choice = ~forced_choice_trials
      bp_values[p] = (transitions_CR & outcomes & ~free_choice) * 1
    elif p == 'cnr_nf':  #unitary predictor - common-nonrew-forced
      free_choice = ~forced_choice_trials
      bp_values[p] = (transitions_CR & ~outcomes & ~free_choice) * 1
    elif p == 'rr_f':  #unitary predictor - rare-rew-free
      free_choice = ~forced_choice_trials
      bp_values[p] = (~transitions_CR & outcomes & free_choice) * 1
    elif p == 'rnr_f':  #unitary predictor - rare-nonrew-free
      free_choice = ~forced_choice_trials
      bp_values[p] = (~transitions_CR & ~outcomes & free_choice) * 1
    elif p == 'rr_nf':  #unitary predictor - rare-rew-forced
      free_choice = ~forced_choice_trials
      bp_values[p] = (~transitions_CR & outcomes & ~free_choice) * 1
    elif p == 'rnr_nf':  #unitary predictor - rare-nonrew-free
      free_choice = ~forced_choice_trials
      bp_values[p] = (~transitions_CR & ~outcomes & ~free_choice) * 1

    elif p == 'cr':  #unitary predictor - common-rew
      bp_values[p] = (transitions_CR & outcomes) * 1
    elif p == 'cnr':  #unitary predictor - common-nonrew
      bp_values[p] = (transitions_CR & ~outcomes) * 1
    elif p == 'rr':  #unitary predictor - rare-rew
      bp_values[p] = (~transitions_CR & outcomes) * 1
    elif p == 'rnr':  #unitary predictor - rare-nonrew-free
      bp_values[p] = (~transitions_CR & ~outcomes) * 1

    elif p == 'correct_choice':  # +0.5 correct choice, -0.5 incorrect choice, 0 in neutral blocks
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = np.asarray([0 if rs ==2 else 0.5 if c == True else -0.5 for c, rs in zip(correct, rew_state)])

    elif p == 'correct_choice_1':  # in the previous trial 1 correct choice, 0 incorrect choice
      bp_values[p] = ((~choices_l1.astype(bool)).astype(int) == (trans_state_l1 ^ rew_state_l1)) * 1

    elif p == 'common_x_correct':  # 0.5 common correct choice, -0.5 common incorrect choice
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      correct = [0.5 if c == True else -0.5 for c in correct]
      bp_values[p] = transitions_CR * correct
    elif p == 'rare_x_correct':  # 0.5 rare correct choice, -0.5 rare incorrect choice
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      correct = [0.5 if c == True else -0.5 for c in correct]
      bp_values[p] = ~transitions_CR * correct

    elif p == 'correct_common' : # 1 common and correct, else 0
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (correct & transitions_CR) * 1
    elif p == 'incorrect_common': # 1 for common and incorrect, else 0
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (~correct & transitions_CR) * 1

    elif p == 'correct_rare' : # 1 rare and correct, else 0
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (correct & ~transitions_CR) * 1
    elif p == 'incorrect_rare': # 1 for rare and incorrect, else 0
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (~correct & ~transitions_CR) * 1

    elif p == 'correct_common_x_rew' : # 1 common and correct, else 0 +1 rew, -1 non_rew
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (correct & transitions_CR) * reward_x
    elif p == 'incorrect_common_x_rew': # 1 for common and incorrect, else 0, else 0 +1 rew, -1 non_rew
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (~correct & transitions_CR) * reward_x

    elif p == 'correct_rare_x_rew' : # 1 rare and correct, else 0, else 0 +1 rew, -1 non_rew
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (correct & ~transitions_CR) * reward_x
    elif p == 'incorrect_rare_x_rew': # 1 for rare and incorrect, else 0, else 0 +1 rew, -1 non_rew
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      bp_values[p] = (~correct & ~transitions_CR) * reward_x

    elif p == 'left':  # 1 left choices, 0 right choices
      bp_values[p] = choices

    elif p == 'rew_state_1': # previous trial, 0.5 if U, -0.5 if D, 0 if N
      bp_values[p] = [0.5 if rs == 1 else -0.5 if rs == 0 else 0 for rs in rew_state_l1]

    elif p == 'common_rare_reward': # common vs rare reward
      same_ss_state = (rew_state == second_steps)*1
      bp_values[p] = [0.5 if ((ss_s == 1) and (o == 1)) else
                      -0.5 if ((ss_s == 0) and (o == 1)) else 0 for ss_s, o in zip(same_ss_state, outcomes)]

    elif p == 'common_rare_unreward': # rare vs rare reward
      same_ss_state = (rew_state == second_steps)*1
      bp_values[p] = [0.5 if ((ss_s == 1) and (o == 0)) else
                      -0.5 if ((ss_s == 0) and (o == 0)) else 0 for ss_s, o in zip(same_ss_state, outcomes)]

    elif p == 'common_rew_unrew': # common reward vs unrews
      same_ss_state = (rew_state == second_steps)*1
      bp_values[p] = [0.5 if ((ss_s == 1) and (o == 1)) else
                      -0.5 if ((ss_s == 1) and (o == 0)) else 0 for ss_s, o in zip(same_ss_state, outcomes)]

    elif p == 'rare_rew_unrew': # rare common reward vs unrews
      same_ss_state = (rew_state == second_steps)*1
      bp_values[p] = [0.5 if ((ss_s == 0) and (o == 1)) else
                      -0.5 if ((ss_s == 0) and (o == 0)) else 0 for ss_s, o in zip(same_ss_state, outcomes)]

    elif p == 'ss_rew_rate':
      mov_rew_rate_up = exp_mov_ave(tau=8, init_value=0.5)
      mov_rew_rate_down = exp_mov_ave(tau=8, init_value=0.5)
      mov_rew_rate_up_session = []
      mov_rew_rate_down_session = []
      for r, ss in zip(outcomes*1, second_steps*1):
        if r == 1 and ss == 0:
          mov_rew_rate_down.update(1)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
          mov_rew_rate_up.update(0.5)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)

        elif r == 0 and ss == 0:
          mov_rew_rate_down.update(0)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
          mov_rew_rate_up.update(0.5)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)

        elif r == 1 and ss == 1:
          mov_rew_rate_up.update(1)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
          mov_rew_rate_down.update(0.5)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)

        elif r == 0 and ss == 1:
          mov_rew_rate_up.update(0)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
          mov_rew_rate_down.update(0.5)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)

      rate_up_l1 = _lag(np.asarray(mov_rew_rate_up_session), 1)
      rate_down_l1 = _lag(np.asarray(mov_rew_rate_down_session), 1)

      all_rate = np.asarray([rate_down_l1, rate_up_l1]).T

      bp_values[p] = [rate[ss] for rate, ss in zip(all_rate, second_steps*1)]

    elif p == 'ss_rew_rate_l1':
      mov_rew_rate_up = exp_mov_ave(tau=8, init_value=0.5)
      mov_rew_rate_down = exp_mov_ave(tau=8, init_value=0.5)
      mov_rew_rate_up_session = []
      mov_rew_rate_down_session = []
      for r, ss in zip(outcomes*1, second_steps*1):
        if r == 1 and ss == 0:
          mov_rew_rate_down.update(1)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
          mov_rew_rate_up.update(0.5)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
        elif r == 0 and ss == 0:
          mov_rew_rate_down.update(0)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
          mov_rew_rate_up.update(0.5)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
        elif r == 1 and ss == 1:
          mov_rew_rate_up.update(1)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
          mov_rew_rate_down.update(0.5)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
        elif r == 0 and ss == 1:
          mov_rew_rate_up.update(0)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
          mov_rew_rate_down.update(0.5)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
      rate_up_l1 = _lag(np.asarray(mov_rew_rate_up_session), 1)
      rate_down_l1 = _lag(np.asarray(mov_rew_rate_down_session), 1)

      all_rate = np.asarray([rate_down_l1, rate_up_l1]).T

      bp_values[p] = [rate[ss] for rate, ss in zip(all_rate, second_steps_l1*1)]

    elif p == 'ss_rew_rate_counterfact':
      mov_rew_rate_up = exp_mov_ave(tau=10, init_value=0.5)
      mov_rew_rate_down = exp_mov_ave(tau=10, init_value=0.5)
      mov_rew_rate_up_session = []
      mov_rew_rate_down_session = []
      for r, ss in zip(outcomes*1, second_steps*1):
        if r == 1 and ss == 0:
          mov_rew_rate_down.update(1)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
          mov_rew_rate_up.update(0)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
        elif r == 0 and ss == 0:
          mov_rew_rate_down.update(0)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
          mov_rew_rate_up.update(1)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
        elif r == 1 and ss == 1:
          mov_rew_rate_up.update(1)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
          mov_rew_rate_down.update(0)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)
        elif r == 0 and ss == 1:
          mov_rew_rate_up.update(0)
          mov_rew_rate_up_session.append(mov_rew_rate_up.value)
          mov_rew_rate_down.update(1)
          mov_rew_rate_down_session.append(mov_rew_rate_down.value)

      rate_up_l1 = _lag(np.asarray(mov_rew_rate_up_session), 1)
      rate_down_l1 = _lag(np.asarray(mov_rew_rate_down_session), 1)

      all_rate = np.asarray([rate_down_l1, rate_up_l1]).T

      bp_values[p] = [rate[ss] for rate, ss in zip(all_rate, second_steps*1)]


    elif p == 'choice_rew_rate':
      mov_rew_rate_left = exp_mov_ave(tau=10, init_value=0.5)
      mov_rew_rate_right = exp_mov_ave(tau=10, init_value=0.5)
      mov_rew_rate_left_session = []
      mov_rew_rate_right_session = []
      for r, ch in zip(outcomes*1, choices*1):
        if r == 1 and ch == 0:
          mov_rew_rate_right.update(1)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)
          mov_rew_rate_left.update(0.5)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)

        elif r == 0 and ch == 0:
          mov_rew_rate_right.update(0)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)
          mov_rew_rate_left.update(0.5)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)

        elif r == 1 and ch == 1:
          mov_rew_rate_left.update(1)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)
          mov_rew_rate_right.update(0.5)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)

        elif r == 0 and ch == 1:
          mov_rew_rate_left.update(0)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)
          mov_rew_rate_right.update(0.5)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)

      rate_left_l1 = _lag(np.asarray(mov_rew_rate_left_session), 1)
      rate_right_l1 = _lag(np.asarray(mov_rew_rate_right_session), 1)

      all_rate = np.asarray([rate_right_l1, rate_left_l1]).T

      bp_values[p] = [rate[ch] for rate, ch in zip(all_rate, choices*1)]

    elif p == 'choice_rew_rate_counterfact':
      mov_rew_rate_left = exp_mov_ave(tau=10, init_value=0.5)
      mov_rew_rate_right = exp_mov_ave(tau=10, init_value=0.5)
      mov_rew_rate_left_session = []
      mov_rew_rate_right_session = []
      for r, ch in zip(outcomes*1, choices*1):
        if r == 1 and ch == 0:
          mov_rew_rate_right.update(1)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)
          mov_rew_rate_left.update(0)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)

        elif r == 0 and ch == 0:
          mov_rew_rate_right.update(0)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)
          mov_rew_rate_left.update(1)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)

        elif r == 1 and ch == 1:
          mov_rew_rate_left.update(1)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)
          mov_rew_rate_right.update(0)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)

        elif r == 0 and ch == 1:
          mov_rew_rate_left.update(0)
          mov_rew_rate_left_session.append(mov_rew_rate_left.value)
          mov_rew_rate_right.update(1)
          mov_rew_rate_right_session.append(mov_rew_rate_right.value)

      rate_left_l1 = _lag(np.asarray(mov_rew_rate_left_session), 1)
      rate_right_l1 = _lag(np.asarray(mov_rew_rate_right_session), 1)

      all_rate = np.asarray([rate_right_l1, rate_left_l1]).T

      bp_values[p] = [rate[ch] for rate, ch in zip(all_rate, choices*1)]


    elif p == 'reward_rate':
      moving_reward_rate = exp_mov_ave(tau=8, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = np.asarray(moving_reward_average_session)
    elif p == 'reward_rate_rolled':
      moving_reward_rate = exp_mov_ave(tau=10, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = np.roll(moving_reward_average_session, 10)

    elif p == 'short_reward_rate':
      moving_reward_rate = exp_mov_ave(tau=5, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = np.asarray(moving_reward_average_session)

    elif p == 'past_reward_rate':
      moving_reward_rate = exp_mov_ave(tau=10, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = _lag(np.asarray(moving_reward_average_session), -5)

    elif p == 'long_reward_rate':
      moving_reward_rate = exp_mov_ave(tau=15, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = np.asarray(moving_reward_average_session)

    elif p == 'n_trials_reward_rate':
      moving_reward_rate = exp_mov_ave(tau=len(choices), init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = np.asarray(moving_reward_average_session)

    elif p == 'correct_rate':
      correct = session_correct(session)
      correct_l1 = _lag(correct, 1)
      moving_correct_rate = exp_mov_ave(tau=10, init_value=0.5)
      moving_correct_average_session = []
      for x in correct_l1:
        moving_correct_rate.update(x)
        moving_correct_rate.update(x)
        moving_correct_average_session.append(moving_correct_rate.value)
      bp_values[p] = np.asarray(moving_correct_average_session)

    elif p == 'reward_rate_binary':
      moving_reward_rate = exp_mov_ave(tau=10, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = np.asarray([-0.5 if x < 0.4 else 0.5 if x > 0.55 else 0 for x in moving_reward_average_session])

    elif p == 'reward_rate_x_reward':  # reward_rate *0.5 if trial rewarded, -0.5 if trial not rewarded
      moving_reward_rate = exp_mov_ave(tau=10, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      bp_values[p] = np.asarray(moving_reward_average_session) * \
                     np.asarray([0.5 if o == True else -0.5 for o in outcomes])

    elif p == 'mov_average':
      mov_ave_1 = _lag(np.asarray(session.trial_data['mov_average']), 1)
      bp_values[p] = np.asarray([-0.5 if ma < 0.4 else 0.5 if ma > 0.7 else 0 for ma in mov_ave_1])

    elif p == 'reward_l1_2':
      reward_l2 = _lag(outcomes, 2)
      reward_l3 = _lag(outcomes, 3)
      bp_values[p] = [1 if (l1==1 and l2==1) else 0 for l1, l2 in zip(reward_l1, reward_l2)]

    elif p == 'same_ssl1_x_rewl1':
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([x * 0.5 if (x == 1 and r1 == 1) else x * (-0.5) if (x == 1 and r1 == 0) else x
                      for x, r1 in zip(same_ss, reward_l1)])
    elif p == 'diff_ssl1_x_rewl1':
      diff_ss = (second_steps != second_steps_l1) * 1
      bp_values[p] = np.asarray([x * 0.5 if (x == 1 and r1 == 1) else x * (-0.5) if (x == 1 and r1 == 0) else x
                      for x, r1 in zip(diff_ss, reward_l1)])

    elif p == 'same_chl1_x_rewl1':
      same_c = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([x * 0.5 if (x == 1 and r1 == 1) else x * (-0.5) if (x == 1 and r1 == 0) else x
                      for x, r1 in zip(same_c, reward_l1)])
    elif p == 'diff_chl1_x_rewl1':
      diff_c = (choices != choices_l1) * 1
      bp_values[p] = np.asarray([x * 0.5 if (x == 1 and r1 == 1) else x * (-0.5) if (x == 1 and r1 == 0) else x
                      for x, r1 in zip(diff_c, reward_l1)])

    elif p == 'same_diff_choice':
      same_c = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if x == 1 else (-0.5) for x in same_c])

    elif p == 'same_choice_rewl1':
      same_c = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x == 1 and r1 == 1) else (-0.5) if (x == 1 and r1 == 0) else 0
                                 for x, r1 in zip(same_c, reward_l1)])
    elif p == 'diff_choice_rewl1':
      same_c = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x == 0 and r1 == 1) else (-0.5) if (x == 0 and r1 == 0) else 0
                                 for x, r1 in zip(same_c, reward_l1)])


    elif p == 'same_diff_ss':
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([0.5 if x == 1 else (-0.5) for x in same_ss])

    elif p == 'second_step_update':  # 'ss_ssl1_x_rewl1' 1 same previous second_step and rewarded / different second_step and nonrewarded
      bp_values[p] = ((second_steps == second_steps_l1) == reward_l1) * 1
    elif p == 'second_step_update_rew':  # 0.5 same previous second_step and rewarded / -0.5 different second_step and rewarded / else 0
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x == 1 and r1 ==1) else (-0.5) if (x == 0 and r1 == 1) else 0
                                 for x, r1 in zip(same_ss, reward_l1)])
    elif p == 'second_step_update_nonrew':  # 0.5 same previous second_step and nonrewarded / -0.5 different second_step and nonrewarded / else 0
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x == 1 and r1 ==0) else (-0.5) if (x == 0 and r1 == 0) else 0
                                 for x, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_same': #  +0.5 sama second-step and rewarded  / -0.5 same second-step non-rewarded / 0 different second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_same_str': #  +0.5 same second-step and rewarded  / -0.5 same second-step non-rewarded / 0 different second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray(['ss_same_pos' if (ss == 1 and r1 == 1) else 'ss_same_neg' if (ss == 1 and r1 == 0) else 'ss_same_0'
                                 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_same_rewonly': #  +0.5 sama second-step and rewarded  / -0.5 same second-step non-rewarded / 0 different second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([1 if (ss == 1 and r1 == 1) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_same_nonrewonly': #  +0.5 sama second-step and rewarded  / -0.5 same second-step non-rewarded / 0 different second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([1 if (ss == 1 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_diff': #  - 0.5 different second-step and non-rewarded  / +0.5 different second-step rewarded / 0 same second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_diff_str': #  - 0.5 different second-step and non-rewarded  / +0.5 different second-step rewarded / 0 same second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray(['ss_dif_pos' if (ss == 0 and r1 == 1) else 'ss_dif_neg' if (ss == 0 and r1 == 0) else 'ss_dif_0'
                                 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_diff_rewonly': #  - 0.5 different second-step and non-rewarded  / +0.5 different second-step rewarded / 0 same second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([0.5 if (ss == 0 and r1 == 1) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
    elif p == 'second_step_update_diff_nonrewonly': #  - 0.5 different second-step and non-rewarded  / +0.5 different second-step rewarded / 0 same second-step
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([0.5 if (ss == 0 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'second_step_update_same_r_x_rew': #  same second_step update interacting with current reward
      same_ss = (second_steps == second_steps_l1) * 1
      same_ss_r1 = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = np.asarray([0.5 if (ssr == 0.5 and r == 1) else -0.5 if (ssr == 0.5 and r == 0) else 0
                                 for ssr, r in zip(same_ss_r1, outcomes * 1)])
    elif p == 'second_step_update_same_nr_x_rew': #  same second_step update interacting with current reward
      same_ss = (second_steps == second_steps_l1) * 1
      same_ss_r1 = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = np.asarray([0.5 if (ssr == -0.5 and r == 1) else -0.5 if (ssr == -0.5 and r == 0) else 0
                                 for ssr, r in zip(same_ss_r1, outcomes * 1)])
    elif p == 'second_step_update_diff_r_x_rew': #  different second_step update interacting with current reward
      same_ss = (second_steps == second_steps_l1) * 1
      diff_ss_r1 = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = np.asarray([0.5 if (dfr == 0.5 and r == 1) else -0.5 if (dfr == 0.5 and r == 0) else 0
                                 for dfr, r in zip(diff_ss_r1, outcomes * 1)])
    elif p == 'second_step_update_diff_nr_x_rew': #  different second_step update interacting with current reward
      same_ss = (second_steps == second_steps_l1) * 1
      diff_ss_r1 = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = np.asarray([0.5 if (dfr == -0.5 and r == 1) else -0.5 if (dfr == -0.5 and r == 0) else 0
                                 for dfr, r in zip(diff_ss_r1, outcomes * 1)])

    elif p == 'second_step_update_same_rew': #  +1 same second-step and rewarded  / else 0
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([1 if (ss == 1 and r1 == 1) else 0 for ss, r1 in zip(same_ss, reward_l1)])
    elif p == 'second_step_update_same_nonrew': #  +1 same second-step and non-rewarded  / else 0
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([1 if (ss == 1 and r1 == 0) else 0 for ss, r1 in zip(same_ss, reward_l1)])
    elif p == 'second_step_update_diff_rew': #  1  different second-step rewarded / else 0
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([1 if (ss == 0 and r1 == 1) else 0 for ss, r1 in zip(same_ss, reward_l1)])
    elif p == 'second_step_update_diff_nonrew': #  1  different second-step non-rewarded / else 0
      same_ss = (second_steps == second_steps_l1) * 1
      bp_values[p] = np.asarray([1 if (ss == 0 and r1 == 0) else 0 for ss, r1 in zip(same_ss, reward_l1)])

    elif p == 'model_free_update':  # +0.5 same previous choice and rewarded / -0.5 same previus choice and non-rewarded/ else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x == 1 and r1 == 1) else -0.5 if (x == 1 and r1 == 0) else 0 for x, r1 in zip(same_ch, reward_l1)])

    elif p == 'model_free_update_str':  # +0.5 same previous choice and rewarded / -0.5 same previus choice and non-rewarded/ else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray(['mf_pos' if (x == 1 and r1 == 1) else 'mf_neg' if (x == 1 and r1 == 0) else 'mf_0' for x, r1 in zip(same_ch, reward_l1)])

    elif p == 'model_free_update_rewonly':  # 1 same previous choice and rewarded  else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([1 if (x == 1 and r1 == 1) else 0 for x, r1 in zip(same_ch, reward_l1)])

    elif p == 'model_free_update_nonrewonly':  # 1 same previous choice and rewarded  else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([1 if (x == 1 and r1 == 0) else 0 for x, r1 in zip(same_ch, reward_l1)])

    elif p == 'model_free_update_rew':  #  0.5 same previous choice and rewarded / -0.5 different previous choice and rewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x ==1 and r1 == 1) else (-0.5) if (x == 0 and r1 == 1) else 0
                                 for x, r1 in zip(same_ch, reward_l1)])
    elif p == 'model_free_update_nonrew':  #  0.5 same previous choice and nonrewarded / -0.5 different previous choice and nonrewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x ==1 and r1 == 0) else (-0.5) if (x == 0 and r1 == 0) else 0
                                 for x, r1 in zip(same_ch, reward_l1)])
    elif p == 'model_free_update_same':  #  0.5 same previous choice and rewarded / -0.5 same previous choice and non-rewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x ==1 and r1 == 1) else (-0.5) if (x == 1 and r1 == 0) else 0
                                 for x, r1 in zip(same_ch, reward_l1)])
    elif p == 'model_free_update_diff':  #  0.5 diff previous choice and rewarded / -0.5 different previous choice and nonrewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if (x ==0 and r1 == 1) else (-0.5) if (x == 0 and r1 == 0) else 0
                                 for x, r1 in zip(same_ch, reward_l1)])

    elif p == 'model_free_update_same_rew':  #  1 same previous choice and rewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([1 if (x == 1 and r1 == 1) else 0 for x, r1 in zip(same_ch, reward_l1)])
    elif p == 'model_free_update_same_nonrew':  #  1 same previous choice and rewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([1 if (x == 1 and r1 == 0) else 0 for x, r1 in zip(same_ch, reward_l1)])
    elif p == 'model_free_update_diff_rew':  #  1 diff previous choice and rewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([1 if (x == 0 and r1 == 1) else 0 for x, r1 in zip(same_ch, reward_l1)])
    elif p == 'model_free_update_diff_nonrew':  #  1 diff previous choice and rewarded / else 0
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([1 if (x == 0 and r1 == 0) else 0 for x, r1 in zip(same_ch, reward_l1)])

    elif p == 'model_based_update':  # 'ch_to_ssl1_x_rewl1' 1 choose poke that commonly leds to the previous second step reached and previously rewarded
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if x == 1 else -0.5 for x in ((choices_to_ssl1 == choices) == reward_l1) * 1])

    elif p == 'model_based_update_rewonly':  # 'ch_to_ssl1_x_rewl1' 1 choose poke that commonly leds to the previous second step reached and previously rewarded
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if (x == c and r1 == 1) else -0.5 if (x != c and r1 == 1) else 0 for x, c, r1 in
                                 zip(choices_to_ssl1, choices, reward_l1)])

    elif p == 'model_based_update_nonrewonly':  # 'ch_to_ssl1_x_rewl1' 1 choose poke that commonly leds to the previous second step reached and previously rewarded
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([-0.5 if (x == c and r1 == 0) else 0.5 if (x != c and r1 == 0) else 0 for x, c, r1 in
                                 zip(choices_to_ssl1, choices, reward_l1)])

    elif p == 'model_based_update__2': # +0.5 choose poke that commonly leads to the previous ss that was rewarded or
      # choose the poke that commonly leads to the opposite ss when the previous trial was unrewarded, otherwise -0.5
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if ((x == 1 and r1 == 1) or (x == 0 and r1 == 0)) else -0.5
                                 for x, r1 in zip(choices_to_ssl1, reward_l1)])

    elif p == 'model_based_update__3': # +0.5 choose the poke that commonly leads to the good ss/ -0.5 choose the poke the commonly leads to the bad ss
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if ((x == rs) and (rs != 2)) else (-0.5) if ((x != rs) and (rs != 2)) else 0
                                 for x, rs in zip(choices_to_ssl1, rew_state)])

    elif p == 'model_based_update_rew':  #  0.5 choose poke that commonly leds to the previous second step reached and previously rewarded
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if (x ==1 and r1 == 1) else (-0.5) if (x == 0 and r1 == 1) else 0
                                 for x, r1 in zip(choices_to_ssl1, reward_l1)])
    elif p == 'model_based_update_nonrew':  #  0.5 choose poke that commonly leds to the previous second step reached and previously nonrewarded
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if (x ==1 and r1 == 0) else (-0.5) if (x == 0 and r1 == 0) else 0
                                 for x, r1 in zip(choices_to_ssl1, reward_l1)])
    elif p == 'model_based_update_same':  #  0.5 choose poke that commonly leds to the previous second step reached and previously rewarded / -0.5 if choose poke that commonly leads to the same previous step and was not rewarded
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if (x ==1 and r1 == 1) else (-0.5) if (x == 1 and r1 == 0) else 0
                                 for x, r1 in zip(choices_to_ssl1, reward_l1)])
    elif p == 'model_based_update_diff':  #  0.5 choose poke that commonly leds to the different previous second step reached and previously rewarded / -0.5 for diff and non-rewarded
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if (x == 0 and r1 == 1) else (-0.5) if (x == 0 and r1 == 0) else 0
                                 for x, r1 in zip(choices_to_ssl1, reward_l1)])

    elif p == 'model_based_update_same_rew':  #  1 choose poke that commonly leds to the previous second step reached and previously rewarded / else 0
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([1 if (x ==1 and r1 == 1) else 0 for x, r1 in zip(choices_to_ssl1, reward_l1)])
    elif p == 'model_based_update_same_nonrew':  #  1 choose poke that commonly leds to the same previous second step reached and previously nonrewarded / else 0
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([0.5 if (x == 1 and r1 == 0) else 0 for x, r1 in zip(choices_to_ssl1, reward_l1)])
    elif p == 'model_based_update_diff_rew':  #  1 choose poke that commonly leds to the different previous second step reached and previously rewarded / else 0
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([1 if (x == 0 and r1 == 1) else 0 for x, r1 in zip(choices_to_ssl1, reward_l1)])
    elif p == 'model_based_update_diff_nonrew':  #  1 choose poke that commonly leds to the diff previous second step reached and previously nonrewarded / else 0
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([1 if (x == 0 and r1 == 0) else 0 for x, r1 in zip(choices_to_ssl1, reward_l1)])

    elif p == 'model_based_update_2':
      second_steps_l2 = _lag(second_steps, 2)
      reward_l2 = _lag(outcomes, 2)
      choices_to_ssl2 = (~(second_steps_l2 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = ((choices_to_ssl2 == choices) == reward_l2) * 1

    elif p == 'model_based_update_3':
      second_steps_l3 = _lag(second_steps, 3)
      reward_l3 = _lag(outcomes, 3)
      choices_to_ssl3 = (~(second_steps_l3 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = ((choices_to_ssl3 == choices) == reward_l3) * 1

    elif p == 'model_based_update_4':
      second_steps_l4 = _lag(second_steps, 4)
      reward_l4 = _lag(outcomes, 4)
      choices_to_ssl4 = (~(second_steps_l4 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = ((choices_to_ssl4 == choices) == reward_l4) * 1

    elif p == 'smf_x_cmb_x_sss_r1':
      bp_values[p] = (same_ch & causal_mb & same_ss) * reward_l1_x

    elif p == 'dmf_x_cmb_x_sss_r1':
      bp_values[p] = (~same_ch & causal_mb & same_ss) * reward_l1_x

    elif p == 'smf_x_ncmb_x_sss_r1':
      bp_values[p] = (same_ch & ~causal_mb & same_ss) * reward_l1_x

    elif p == 'dmf_x_ncmb_x_sss_r1':
      bp_values[p] = (~same_ch & ~causal_mb & same_ss) * reward_l1_x

    elif p == 'smf_x_cmb_x_dss_r1':
      bp_values[p] = (same_ch & causal_mb & ~same_ss) * reward_l1_x

    elif p == 'dmf_x_cmb_x_dss_r1':
      bp_values[p] = (~same_ch & causal_mb & ~same_ss) * reward_l1_x

    elif p == 'smf_x_ncmb_x_dss_r1':
      bp_values[p] = (same_ch & ~causal_mb & ~same_ss) * reward_l1_x

    elif p == 'dmf_x_ncmb_x_dss_r1':
      bp_values[p] = (~same_ch & ~causal_mb & ~same_ss) * reward_l1_x

    elif p == 'sch_x_sss_r1':
      bp_values[p] = (same_ch & same_ss) * reward_l1_x * 0.5

    elif p == 'sch_x_dss_r1':
      bp_values[p] = (same_ch & ~same_ss) * reward_l1_x * 0.5

    elif p == 'dch_x_sss_r1':
      bp_values[p] = (~same_ch & same_ss) * reward_l1_x * 0.5

    elif p == 'dch_x_dss_r1':
      bp_values[p] = (~same_ch & ~same_ss) * reward_l1_x * 0.5

    elif p == 'intercept':
      bp_values[p] = [1 for x in choices]

    elif p == 'left_right': # +0.5 left choices, -0.5 right choices
      bp_values[p] = [0.5 if c == 1 else -0.5 for c in (choices * 1)]

    elif p == 'up_down': # +0.5 up ss, -0.5 down ss
      bp_values[p] = [0.5 if ss == 1 else -0.5 for ss in (second_steps * 1)]

    elif p == 'up_down_rew': # +0.5 up ss, -0.5 down ss only rew
      bp_values[p] = [0.5 if ((ss == 1) and (o == 1)) else -0.5 if ((ss == 0) and (o == 1)) else 0
                      for ss, o in zip(second_steps * 1, outcomes * 1)]

    elif p == 'up_down_unrew': # +0.5 up ss, -0.5 down ss only unrew
      bp_values[p] = [0.5 if ((ss == 1) and (o == 0)) else -0.5 if ((ss == 0) and (o == 0)) else 0
                      for ss, o in zip(second_steps * 1, outcomes * 1)]

    elif p == 'ipsi_contra_choice': # +0.5 ipsilateral choice, -0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      bp_values[p] = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]

    elif p == 'contralateral_choice': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      bp_values[p] = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]

    elif p == 'contralateral_choice_model_free': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      model_free = ((choices == choices_l1) == reward_l1)
      bp_values[p] = [mf if cont == 0.5 else 0 for mf, cont in zip(model_free, contra_ch)]
    elif p == 'ipsi_choice_model_free': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      model_free = ((choices == choices_l1) == reward_l1)
      bp_values[p] = [mf if cont == - 0.5 else 0 for mf, cont in zip(model_free, contra_ch)]
    elif p == 'contralateral_choice_model_based': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      model_based = ((choices_to_ssl1 == choices) == reward_l1)
      bp_values[p] = [mf if cont == 0.5 else 0 for mf, cont in zip(model_based, contra_ch)]
    elif p == 'ipsi_choice_model_based': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      model_based = ((choices_to_ssl1 == choices) == reward_l1)
      bp_values[p] = [mf if cont == -0.5 else 0 for mf, cont in zip(model_based, contra_ch)]
    elif p == 'ipsi_choice_ss_update_same': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      same_ss = (second_steps == second_steps_l1) * 1
      ss_update = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = [upd if cont == -0.5 else 0 for upd, cont in zip(ss_update, contra_ch)]
    elif p == 'ipsi_choice_ss_update_diff': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      same_ss = (second_steps == second_steps_l1) * 1
      ss_update = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = [upd if cont == -0.5 else 0 for upd, cont in zip(ss_update, contra_ch)]
    elif p == 'contra_choice_ss_update_same': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      same_ss = (second_steps == second_steps_l1) * 1
      ss_update = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = [upd if cont == 0.5 else 0 for upd, cont in zip(ss_update, contra_ch)]
    elif p == 'contra_choice_ss_update_diff': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      same_ss = (second_steps == second_steps_l1) * 1
      ss_update = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])
      bp_values[p] = [upd if cont == 0.5 else 0 for upd, cont in zip(ss_update, contra_ch)]
    elif p == 'contra_choice_correct': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      correct_upd = np.asarray([0 if rs == 2 else 0.5 if c == True else -0.5 for c, rs in zip(correct, rew_state)])
      bp_values[p] = [upd if cont == 0.5 else 0 for upd, cont in zip(correct_upd, contra_ch)]
    elif p == 'ipsi_choice_correct': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
      correct_upd = np.asarray([0 if rs == 2 else 0.5 if c == True else -0.5 for c, rs in zip(correct, rew_state)])
      bp_values[p] = [upd if cont == -0.5 else 0 for upd, cont in zip(correct_upd, contra_ch)]


    elif p == 'ipsi_contra_model_free': # +0.5 ipsilateral model_free update, -0.5 contralateral model_free_update
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      bp_values[p] = ((choices == choices_l1) == reward_l1) * hemisphere_param

    elif p == 'ipsi_contra_model_based': # +0.5 ipsilateral model_based update, -0.5 contralateral model_based_update
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = ((choices_to_ssl1 == choices) == reward_l1) * hemisphere_param

    elif p == 'ipsi_contra_choice_reward': # +0.5 ipsi reward -0.5 contra reward
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      bp_values[p] = outcomes * [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]
    elif p == 'ipsi_contra_choice_nonreward': # +0.5 ipsi reward -0.5 contra reward
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      bp_values[p] = ~outcomes * [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]

    elif p == 'ipsi_contra_choice_common_reward': # +0.5 ipsi reward -0.5 contra reward
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      bp_values[p] = transition_CR_x_outcome * outcomes * [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]
    elif p == 'ipsi_contra_choice_common_nonreward': # +0.5 ipsi reward -0.5 contra reward
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      bp_values[p] = transition_CR_x_outcome * ~outcomes * [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]

    elif p == 'ipsi_contra_choice_rare_reward': # +0.5 ipsi reward -0.5 contra reward
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      bp_values[p] = ~transition_CR_x_outcome * outcomes * [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]
    elif p == 'ipsi_contra_choice_rare_nonreward': # +0.5 ipsi reward -0.5 contra reward
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      bp_values[p] = ~transition_CR_x_outcome * ~outcomes * [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]

    elif p == 'ipsi_contra_choice_good_ss':
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      good_bad = np.asarray([1 if (ss == rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_bad *  [
        0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]
    elif p == 'ipsi_contra_choice_bad_ss':
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      good_bad = np.asarray([1 if (ss != rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_bad  * [
        0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]

    elif p == 'ipsi_contra_choice_good_ss_rew':
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      good_bad = np.asarray([1 if (ss == rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_bad * outcomes * [
        0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]
    elif p == 'ipsi_contra_choice_bad_ss_rew':
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      good_bad = np.asarray([1 if (ss != rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_bad * outcomes * [
        0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]
    elif p == 'ipsi_contra_choice_good_ss_nonrew':
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      good_bad = np.asarray([1 if (ss == rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_bad * ~outcomes * [
        0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]
    elif p == 'ipsi_contra_choice_bad_ss_nonrew':
      hemisphere_param = [0.5 if hemisphere=='L' else -0.5][0]
      good_bad = np.asarray([1 if (ss != rs and rs != 2) else 0
                                 for ss, rs in zip(second_steps, rew_state)])
      bp_values[p] = good_bad * ~outcomes * [
        0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)]

    elif p == 'choice_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choice_state', 'choose_right', 'choose_left']])
      times_start = [session.events[x][0] for x in all_id[::2]]
      times_choice = [session.events[x][0] for x in all_id[1::2]]
      # bp_values[p] = np.asarray(times_choice) - np.asarray(times_start)
      bp_values[p] = [e2 - e1 for e1, e2 in zip(times_start, times_choice)][:len(choices)]
    elif p == 'ss_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choose_left', 'choose_right', 'choose_up', 'choose_down']])
      times_choice = [session.events[x][0] for x in all_id[::2]]
      times_ss = [session.events[x][0] for x in all_id[1::2]]
      bp_values[p] = [e2 - e1 for e1, e2 in zip(times_choice, times_ss)][:len(choices)]

    elif p == 'median_choice_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choice_state', 'choose_right', 'choose_left']])
      times_start = [session.events[x][0] for x in all_id[::2]]
      times_choice = [session.events[x][0] for x in all_id[1::2]]
      choice_latency = [e2 - e1 for e1, e2 in zip(times_start, times_choice)][:len(choices)]
      bp_values[p] = [0.5 if cl <= np.median(choice_latency) else -0.5 for cl in choice_latency]

    elif p == 'median_ss_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choose_left', 'choose_right', 'choose_up',
                                                                 'choose_down']])
      times_choice = [session.events[x][0] for x in all_id[::2]]
      times_ss = [session.events[x][0] for x in all_id[1::2]]
      ss_latency = [e2 - e1 for e1, e2 in zip(times_choice, times_ss)][:len(choices)]
      bp_values[p] = [0.5 if cl <= np.median(ss_latency) else -0.5 for cl in ss_latency]

    elif p == 'perc_choice_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choice_state', 'choose_right', 'choose_left']])
      times_start = [session.events[x][0] for x in all_id[::2]]
      times_choice = [session.events[x][0] for x in all_id[1::2]]
      choice_latency = [e2 - e1 for e1, e2 in zip(times_start, times_choice)][:len(choices)]
      bp_values[p] = [0.5 if cl <= np.percentile(choice_latency, 25) else
                      -0.5 if cl >= np.percentile(choice_latency, 75) else 0 for cl in choice_latency]

    elif p == 'perc_ss_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choose_left', 'choose_right', 'choose_up',
                                                                 'choose_down']])
      times_choice = [session.events[x][0] for x in all_id[::2]]
      times_ss = [session.events[x][0] for x in all_id[1::2]]
      ss_latency = [e2 - e1 for e1, e2 in zip(times_choice, times_ss)][:len(choices)]
      bp_values[p] = [0.5 if sl <= np.percentile(ss_latency, 25) else
                      -0.5 if sl >= np.percentile(ss_latency, 75) else 0 for sl in ss_latency]

    elif p == 'perc25_choice_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choice_state', 'choose_right', 'choose_left']])
      times_start = [session.events[x][0] for x in all_id[::2]]
      times_choice = [session.events[x][0] for x in all_id[1::2]]
      choice_latency = [e2 - e1 for e1, e2 in zip(times_start, times_choice)][:len(choices)]
      bp_values[p] = [0.5 if cl <= np.percentile(choice_latency, 25) else -0.5 for cl in choice_latency]

    elif p == 'perc25_ss_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choose_left', 'choose_right', 'choose_up',
                                                                 'choose_down']])
      times_choice = [session.events[x][0] for x in all_id[::2]]
      times_ss = [session.events[x][0] for x in all_id[1::2]]
      ss_latency = [e2 - e1 for e1, e2 in zip(times_choice, times_ss)][:len(choices)]
      bp_values[p] = [0.5 if sl <= np.percentile(ss_latency, 25) else -0.5 for sl in ss_latency]

    elif p == 'log_choice_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choice_state', 'choose_right', 'choose_left']])
      times_start = [session.events[x][0] for x in all_id[::2]]
      times_choice = [session.events[x][0] for x in all_id[1::2]]
      choice_latency = [e2 - e1 for e1, e2 in zip(times_start, times_choice)][:len(choices)]
      bp_values[p] = list(np.log(choice_latency))

    elif p == 'log_start_leave_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choice_state', 'poke_5_out']])
      all_id = ([0] + [i for i in range(1, len(all_id_names)) if all_id_names[i] != all_id_names[i-1]])
      times_start = [session.events[x][0] for x in all_id[::2]]
      times_choice = [session.events[x][0] for x in all_id[1::2]]
      bp_values[p] = ([np.log(e2 - e1) if e2 - e1 != 0 else 0 for e1, e2 in zip(times_start, times_choice)][:len(choices)])

    elif p == 'start_leave_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choice_state', 'poke_5_out']])
      all_id = ([0] + [i for i in range(1, len(all_id_names)) if all_id_names[i] != all_id_names[i-1]])
      times_start = [session.events[x][0] for x in all_id[::2]]
      times_choice = [session.events[x][0] for x in all_id[1::2]]
      choice_latency = [e2 - e1 for e1, e2 in zip(times_start, times_choice)][:len(choices)]
      bp_values[p] = choice_latency

    elif p == 'log_ss_latency':
      all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                   if session.events[i].name in ['choose_left', 'choose_right', 'choose_up',
                                                                 'choose_down']])
      times_choice = [session.events[x][0] for x in all_id[::2]]
      times_ss = [session.events[x][0] for x in all_id[1::2]]
      ss_latency = [e2 - e1 for e1, e2 in zip(times_choice, times_ss)][:len(choices)]
      bp_values[p] = list(np.log(ss_latency))

    elif p == 'ITI_1':
      lat = session_latencies(session, 'ITI')
      if len(lat) < session.n_trials:
        lat = np.append(lat, np.nan)
      lag_lat = _lag(lat, 1)
      bp_values[p] = [(x-min(lag_lat)) / (max(lag_lat)-min(lag_lat)) for x in lag_lat]

    elif p == 'ITI_choice_1':
      lat = session_latencies(session, 'ITI-choice')
      if len(lat) < session.n_trials:
        lat = np.append(lat, np.nan)
      bp_values[p] = _lag(lat, 1)

    elif p == 'ITI_start_1':
      lat = session_latencies(session, 'ITI-start')
      if len(lat) < session.n_trials:
        lat = np.append(lat, np.nan)
      bp_values[p] = _lag(lat, 1)

    elif p== 'subject_ID':
      bp_values[p] = [str(session.subject_ID) for i in range(len(choices))]

    elif p == 'repeat_choice': # +0.5 if same choice as previous choice, -0.5 if different choice to previous trial
      same_ch = (choices == choices_l1) * 1
      bp_values[p] = np.asarray([0.5 if c == 1 else -0.5 for c in same_ch])

    elif p == 'neutral_state': #  -0.5 if end block neutral, +0.5 if start block neutral
      bp_values[p] = session.select_trials(selection_type='end', select_n=15, block_type='neutral') * -0.5 + \
                     session.select_trials(selection_type='start', select_n=15, block_type='neutral') * 0.5

    elif p == 'block_nn': #  -0.5 if end block non_neutral, +0.5 if start block non_neutral
      bp_values[p] = session.select_trials(selection_type='end', select_n=15, block_type='non_neutral') * -0.5 + \
                     session.select_trials(selection_type='start', select_n=15, block_type='non_neutral') * 0.5

    elif p == 'start_end_block':
      bp_values[p] = session.select_trials(selection_type='end', select_n=15, block_type='all') * -0.5 + \
                     session.select_trials(selection_type='start', select_n=15, block_type='all') * 0.5

    elif p == 'block_type':
      bp_values[p] = session.select_trials(selection_type='all', select_n=15, block_type='neutral') * -0.5 + \
                     session.select_trials(selection_type='all', select_n=15, block_type='non_neutral') * 0.5

    elif p == 'neutral':
      bp_values[p] = session.select_trials(selection_type='all', select_n=15, block_type='neutral') * 1

    elif p == 'non_neutral_after_non_neutral':
      bp_values[p] = session.select_trials(selection_type='all', select_n=15, block_type='non_neutral_after_non_neutral') * 1
    elif p == 'non_neutral_after_neutral':
      bp_values[p] = session.select_trials(selection_type='all', select_n=15, block_type='non_neutral_after_neutral') * 1

    elif p == 'non_neutral_after_non_neutral_start_end':
      bp_values[p] = session.select_trials(selection_type='start', select_n=15, block_type='non_neutral_after_non_neutral') * 1
    elif p == 'non_neutral_after_neutral_start_end':
      bp_values[p] = session.select_trials(selection_type='start', select_n=15, block_type='non_neutral_after_neutral') * 1

    elif p == 'Q_net':
      non_choice = (~choices).astype(int)
      bp_values[p] = np.asarray([q[c] - q[nc] for q, c, nc in zip(Q_values['Q_net'], choices * 1, non_choice)])
    elif p == 'Q_net_ch':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_net'], choices * 1)])
    elif p == 'Q_net_nch':
      non_choice = (~choices).astype(int)
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_net'], non_choice * 1)])
    elif p == 'Q_net_ch_1':
      q_l1 = _lag(Q_values['Q_net'], 1)
      bp_values[p] = np.asarray([q[c] for q, c in zip(q_l1, choices_l1 * 1)])
    elif p == 'Q_net_Q_mb_ch':
      bp_values[p] = np.asarray([q[c] - qmb[c] for q, qmb, c in zip(Q_values['Q_net'], Q_values['Q_mb'], choices * 1)])
    elif p == 'Q_mb_update_ch':
      q_f1 = _lag(Q_values['Q_mb'], -1)
      bp_values[p] = np.asarray([q[c] - qmb[c] for q, qmb, c in zip(q_f1, Q_values['Q_mb'], choices * 1)])
    elif p == 'Q_mb':
      bp_values[p] = Q_values['Q_mb'].T[0]
    elif p == 'Q_mb_ch':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_mb'], choices * 1)])
    elif p == 'Q_mb_uncorr_ch':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_mb_uncorr'], choices * 1)])
    elif p == 'Q_mb_uncorr_good':
      best_val = (np.asarray([q[1] - q[0] for q in np.asarray(Q_values['Q_mb_uncorr'])]) > 0).astype(int)
      bp_values[p] = np.asarray([q[b] for q, b in zip(Q_values['Q_mb_uncorr'], best_val)])
    elif p == 'Q_mb_uncorr_ch_w':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_mb_uncorr'], choices * 1)]) * fits[fits_names.index('weight_mb_uncorr')]
    elif p == 'Q_mb_corr_ch':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_mb_corr'], choices * 1)])
    elif p == 'Q_mb_corr_good':
      best_val = (np.asarray([q[1] - q[0] for q in np.asarray(Q_values['Q_mb_corr'])]) > 0).astype(int)
      bp_values[p] = np.asarray([q[b] for q, b in zip(Q_values['Q_mb_corr'], best_val)])
    elif p == 'Q_mb_corr_ch_ipsi':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray([q[c] if cont == -0.5 else 0 for q, c, cont in zip(Q_values['Q_mb_corr'], choices * 1, contra_ch)])
    elif p == 'Q_mb_corr_ch_contra':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray([q[c] if cont == 0.5 else 0 for q, c, cont in zip(Q_values['Q_mb_corr'], choices * 1, contra_ch)])
    elif p == 'Q_mb_bayes_ch':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_mb'], choices * 1)]) / fits[fits_names.index('weight_inf')]
    elif p == 'Q_mb_bayes_ch_stand':
      bp_values[p] = stats.zscore(np.asarray([q[c] for q, c in zip(Q_values['Q_mb'], choices * 1)]) / fits[fits_names.index('weight_inf')])
    elif p == 'Q_mb_bayes_ch_ipsi':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray(
        [q[c] if cont == -0.5 else 0 for q, c, cont in zip(Q_values['Q_mb'], choices * 1, contra_ch)]) / fits[
                       fits_names.index('weight_inf')]
    elif p == 'Q_mb_bayes_ch_contra':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray([q[c] if cont == 0.5 else 0 for q, c, cont in zip(Q_values['Q_mb'], choices * 1, contra_ch)]) / fits[fits_names.index('weight_inf')]
    elif p == 'Q_mb_bayes_good':
      best_val = (np.asarray([q[1] - q[0] for q in np.asarray(Q_values['Q_mb']) / fits[fits_names.index('weight_inf')]]) > 0).astype(int)
      bp_values[p] = np.asarray([q[b] for q, b in zip(Q_values['Q_mb'], best_val)]) / fits[fits_names.index('weight_inf')]
    elif p == 'Q_mb_bayes_ch_forced':
      bp_values[p] = np.asarray([q[c] if f == 1 else 0 for q, c, f in
                                 zip(Q_values['Q_mb'], choices * 1, forced_choice_trials * 1)]) / fits[fits_names.index('weight_inf')]
    elif p == 'Q_mb_bayes_ch_free':
      bp_values[p] = np.asarray([q[c] if f == 0 else 0 for q, c, f in
                                 zip(Q_values['Q_mb'], choices * 1, forced_choice_trials * 1)]) / fits[fits_names.index('weight_inf')]
    elif p == 'Q_mb_bayes_ch_free_forced':
      bp_values[p] = np.asarray([q[c] if f == 0 else -q[c] for q, c, f in
                                 zip(Q_values['Q_mb'], choices * 1, forced_choice_trials * 1)]) / fits[fits_names.index('weight_inf')]
    elif p == 'Q_mb_bayes_ch_from_v':
      choices_com_ss = (~(choices ^ trans_state).astype(bool)).astype(int)
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['V_s'], choices_com_ss)])
    elif p == 'Q_mb_bayes':
      bp_values[p] = np.asarray([q[1] for q in Q_values['Q_mb']])
    elif p == 'Q_mb_corr_ch_w':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_mb_corr'], choices * 1)])* fits[fits_names.index('weight_mb_corr')]
    elif p == 'Q_mb_ch_rew':
      bp_values[p] = np.asarray([q[c] if o == 1 else 0 for q, c, o in zip(Q_values['Q_mb'], choices * 1, outcomes * 1)])
    elif p == 'Q_mb_ch_unrew':
      bp_values[p] = np.asarray([q[c] if o == 0 else 0 for q, c, o in zip(Q_values['Q_mb'], choices * 1, outcomes * 1)])
    elif p == 'Q_mb_ch_1':
      q_l1 = _lag(Q_values['Q_mb'], 1)
      bp_values[p] = np.asarray([q[c] for q, c in zip(q_l1, choices_l1 * 1)])
    elif p == 'Q_mb_ch_future':
      future_Q = _lag(Q_values['Q_mb'], -1)
      bp_values[p] = np.asarray([q[c] for q, c in zip(future_Q, choices * 1)])
    elif p == 'Q_mb_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['Q_mb'], second_steps * 1)])
    elif p == 'Q_mf':
      if 'Q_c' in Q_values:
        bp_values[p] = Q_values['Q_c'].T[0]
      else:
        bp_values[p] = Q_values['Q_c_mf'].T[0]
    elif p == 'Q_mf_ch':
      if 'Q_c' in Q_values:
        bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_c'], choices * 1)])
      else:
        bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_c_mf'], choices * 1)])
    elif p == 'Q_mf_ch_rew':
        if 'Q_c' in Q_values:
          bp_values[p] = np.asarray([q[c] if o == 1 else 0 for q, c, o in zip(Q_values['Q_c'], choices * 1, outcomes * 1)])
        else:
          bp_values[p] = np.asarray([q[c] if o == 1 else 0 for q, c, o in zip(Q_values['Q_c_mf'], choices * 1, outcomes * 1)])
    elif p == 'Q_mf_ch_unrew':
      if 'Q_c' in Q_values:
        bp_values[p] = np.asarray([q[c] if o == 0 else 0 for q, c, o in zip(Q_values['Q_c'], choices * 1, outcomes * 1)])
      else:
        bp_values[p] = np.asarray([q[c] if o == 0 else 0 for q, c, o in zip(Q_values['Q_c_mf'], choices * 1, outcomes * 1)])
    elif p == 'Q_mf_ch_w':
      if 'Q_c' in Q_values:
        bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_c'], choices * 1)]) * fits[fits_names.index('weight_mf')]
    elif p == 'Q_mf_ch_1':
      q_l1 = _lag(Q_values['Q_c'], 1)
      bp_values[p] = np.asarray([q[c] for q, c in zip(q_l1, choices_l1 * 1)])
    elif p == 'Q_mf_ch_future':
      q_l1 = _lag(Q_values['Q_c'], -1)
      bp_values[p] = np.asarray([q[c] for q, c in zip(q_l1, choices_l1 * 1)])
    elif p == 'Q_mf_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['Q_c'], second_steps * 1)])
    elif p == 'V_s':
      trans_type = trans_state[0] * 1
      bp_values[p] = Q_values['V_s'].T[trans_type]
    elif p == 'V_s_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
    elif p == 'V_s_ss_stand':
      bp_values[p] = stats.zscore(np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)]))
    elif p == 'V_s_actual_minus_causal':
      choices_com_ss = (~(choices ^ trans_state).astype(bool)).astype(int)
      value_causal = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], choices_com_ss)])
      value_actual = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
      bp_values[p] = np.asarray([va - vc for va, vc in zip(value_actual, value_causal)])
    elif p == 'Q_mb_bayes_ch_v_ss':
      v_s = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
      q_s = np.asarray([q[c]/fits[fits_names.index('weight_inf')] for q, c in zip(Q_values['Q_mb'], choices * 1)])
      bp_values[p] = np.asarray([v - q for v, q in zip(v_s, q_s)])
    elif p == 'V_s_uncorr_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['V_s_uncorr'], second_steps * 1)])
    elif p == 'V_s_corr_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['V_s_corr'], second_steps * 1)])
    elif p == 'V_s_corr_ss_ipsi':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray([q[ss] if cont==-0.5 else 0 for q, ss, cont in zip(Q_values['V_s_corr'], second_steps * 1, contra_ch)])
    elif p == 'V_s_corr_ss_contra':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray([q[ss] if cont==0.5 else 0 for q, ss, cont in zip(Q_values['V_s_corr'], second_steps * 1, contra_ch)])
    elif p == 'V_s_ss_ipsi':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray(
        [q[ss] if cont == -0.5 else 0 for q, ss, cont in zip(Q_values['V_s'], second_steps * 1, contra_ch)])
    elif p == 'V_s_ss_contra':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      bp_values[p] = np.asarray(
        [q[ss] if cont == 0.5 else 0 for q, ss, cont in zip(Q_values['V_s'], second_steps * 1, contra_ch)])
    elif p == 'V_s_bayes_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
    elif p == 'V_s_bayes':
      bp_values[p] = np.asarray([q[1] for q in Q_values['V_s']])
    elif p == 'V_s_bayes_before_pr':
      bp_values[p] = np.asarray([q[1] for q in Q_values['V_s']])
    elif p == 'V_s_ss_rew':
      bp_values[p] = np.asarray([q[ss] if o == 1 else 0 for q, ss, o in zip(Q_values['V_s'], second_steps * 1, outcomes * 1)])
    elif p == 'V_s_ss_unrew':
      bp_values[p] = np.asarray([q[ss] if o == 0 else 0 for q, ss, o in zip(Q_values['V_s'], second_steps * 1, outcomes * 1)])
    elif p == 'V_s_uncorr_ss_rew':
      bp_values[p] = np.asarray([q[ss] if o == 1 else 0 for q, ss, o in zip(Q_values['V_s_uncorr'], second_steps * 1, outcomes * 1)])
    elif p == 'V_s_uncorr_ss_unrew':
      bp_values[p] = np.asarray([q[ss] if o == 0 else 0 for q, ss, o in zip(Q_values['V_s_uncorr'], second_steps * 1, outcomes * 1)])
    elif p == 'V_s_corr_ss_rew':
      bp_values[p] = np.asarray([q[ss] if o == 1 else 0 for q, ss, o in zip(Q_values['V_s_corr'], second_steps * 1, outcomes * 1)])
    elif p == 'V_s_corr_ss_unrew':
      bp_values[p] = np.asarray([q[ss] if o == 0 else 0 for q, ss, o in zip(Q_values['V_s_corr'], second_steps * 1, outcomes * 1)])
    elif p == 'V_s_mf_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['V_s_mf'], second_steps * 1)])
    elif p == 'V_s_mb_ss':
      bp_values[p] = np.asarray([q[ss] for q, ss in zip(Q_values['V_s_mb'], second_steps * 1)])

    elif p == 'V_s_mf_ss_rew':
      bp_values[p] = np.asarray([q[ss] if r1 == 1 else 0 for q, ss, r1 in zip(Q_values['V_s_mf'], second_steps * 1, reward_l1)])
    elif p == 'V_s_mb_ss_rew':
      bp_values[p] = np.asarray([q[ss] if r1 == 1 else 0 for q, ss, r1 in zip(Q_values['V_s_mb'], second_steps * 1, reward_l1)])

    elif p == 'V_s_mf_ss_nonrew':
      bp_values[p] = np.asarray([q[ss] if r1 == 0 else 0 for q, ss, r1 in zip(Q_values['V_s_mf'], second_steps * 1, reward_l1)])
    elif p == 'V_s_mb_ss_nonrew':
      bp_values[p] = np.asarray([q[ss] if r1 == 0 else 0 for q, ss, r1 in zip(Q_values['V_s_mb'], second_steps * 1, reward_l1)])

    elif p == 'diff_V_s':
      bp_values[p] = np.abs(Q_values['V_s'].T[0] - Q_values['V_s'].T[1])
    elif p == 'diff_Q_mb':
      bp_values[p] = np.abs(Q_values['Q_mb'].T[0]/ fits[fits_names.index('weight_inf')] - Q_values['Q_mb'].T[1] / fits[fits_names.index('weight_inf')])

    elif p == 'diff_Q_mb_ipsi':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      diff_val = np.abs(Q_values['Q_mb'].T[0]/ fits[fits_names.index('weight_inf')] - Q_values['Q_mb'].T[1]/ fits[fits_names.index('weight_inf')])
      bp_values[p] = np.asarray([val if cont==-0.5 else 0 for val, cont in zip(diff_val, contra_ch)])
    elif p == 'diff_Q_mb_contra':
      hemisphere_param = [1 if hemisphere == 'L' else -1][0]
      contra_ch = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)]
      diff_val = np.abs(Q_values['Q_mb'].T[0]/ fits[fits_names.index('weight_inf')] - Q_values['Q_mb'].T[1]/ fits[fits_names.index('weight_inf')])
      bp_values[p] = np.asarray([val if cont==0.5 else 0 for val, cont in zip(diff_val, contra_ch)])

    elif p == 'Q_hyb_ch':
      bp_values[p] = np.asarray([q[c] for q, c in zip(Q_values['Q_hyb'], choices * 1)])
    elif p == 'Q_hyb_ch_1':
      q_l1 = _lag(Q_values['Q_hyb'], 1)
      bp_values[p] = np.asarray([q[c] for q, c in zip(q_l1, choices_l1 * 1)])

    elif p == 'RPE_ss':
      v_s = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
      v_s_1 = np.asarray([q[ss] for q, ss in zip(_lag(Q_values['V_s'], -1), second_steps * 1)])
      bp_values[p] = np.asarray([r + v1 - v for r, v, v1 in zip(outcomes, v_s, v_s_1)])
    elif p == 'RPE_ss_start':
      v_s = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
      v_s_1 = np.asarray([q[ss] for q, ss in zip(_lag(Q_values['V_s'], -1), second_steps * 1)])
      select_start = session.select_trials(selection_type='start', select_n=15, block_type='all')
      rpe = np.asarray([r + v - v1 for r, v, v1 in zip(outcomes, v_s, v_s_1)])
      bp_values[p] = np.asarray([e if s == True else 0 for e, s in zip(rpe, select_start)])
    elif p == 'RPE_ss_end':
      v_s = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
      v_s_1 = np.asarray([q[ss] for q, ss in zip(_lag(Q_values['V_s'], -1), second_steps * 1)])
      select_end = session.select_trials(selection_type='end', select_n=15, block_type='all')
      rpe = np.asarray([r + v - v1 for r, v, v1 in zip(outcomes, v_s, v_s_1)])
      bp_values[p] = np.asarray([e if s == True else 0 for e, s in zip(rpe, select_end)])
    elif p == 'RPE_mf_ch':
      reward = [1 if r==1 else -1 for r in (outcomes * 1)]
      v_ch = np.asarray([q[c] for q, c in zip(Q_values['Q_c'], choices * 1)])
      v_ch_1 = np.asarray([q[c] for q, c in zip(_lag(Q_values['Q_c'], -1), choices * 1)])
      bp_values[p] = np.asarray([v - v1 for r, v, v1 in zip(outcomes, v_ch, v_ch_1)])
    elif p == 'RPE_mb_ch':
      reward = [1 if r==1 else -1 for r in (outcomes * 1)]
      v_ch = np.asarray([q[c] for q, c in zip(Q_values['Q_mb'], choices * 1)])
      v_ch_1 = np.asarray([q[c] for q, c in zip(_lag(Q_values['Q_mb'], -1), choices * 1)])
      bp_values[p] = np.asarray([r + v - v1 for r, v, v1 in zip(reward, v_ch, v_ch_1)])
    elif p == 'RPE_hyb_ch':
      v_ch = np.asarray([q[c] for q, c in zip(Q_values['Q_hyb'], choices * 1)])
      v_ch_1 = np.asarray([q[c] for q, c in zip(_lag(Q_values['Q_hyb'], -1), choices * 1)])
      bp_values[p] = np.asarray([r + v - v1 for r, v, v1 in zip(outcomes, v_ch, v_ch_1)])
    elif p == 'RPE_mb_ch_start':
      v_ch = np.asarray([q[c] for q, c in zip(Q_values['Q_mb'], choices * 1)])
      v_ch_1 = np.asarray([q[c] for q, c in zip(_lag(Q_values['Q_mb'], -1), choices * 1)])
      select_start = session.select_trials(selection_type='start', select_n=15, block_type='all')
      rpe = np.asarray([r + v - v1 for r, v, v1 in zip(outcomes, v_ch, v_ch_1)])
      bp_values[p] = np.asarray([e if s == True else 0 for e, s in zip(rpe, select_start)])
    elif p == 'RPE_mb_ch_end':
      v_ch = np.asarray([q[c] for q, c in zip(Q_values['Q_mb'], choices * 1)])
      v_ch_1 = np.asarray([q[c] for q, c in zip(_lag(Q_values['Q_mb'], -1), choices * 1)])
      select_end = session.select_trials(selection_type='end', select_n=15, block_type='all')
      rpe = np.asarray([r + v - v1 for r, v, v1 in zip(outcomes, v_ch, v_ch_1)])
      bp_values[p] = np.asarray([e if s == True else 0 for e, s in zip(rpe, select_end)])
    elif p == 'RPE_Q_net_ch':
      v_ch = np.asarray([q[c] for q, c in zip(Q_values['Q_net'], choices * 1)])
      v_ch_1 = np.asarray([q[c] for q, c in zip(_lag(Q_values['Q_net'], -1), choices * 1)])
      bp_values[p] = np.asarray([v - v1 for r, v, v1 in zip(outcomes, v_ch, v_ch_1)])
    elif p == 'RPE_mf_ch_ss':
      v_ch = np.asarray([q[c] for q, c in zip(Q_values['Q_c'], choices * 1)])
      v_s = np.asarray([q[ss] for q, ss in zip(Q_values['V_s'], second_steps * 1)])
      bp_values[p] = np.asarray([v - v1 for v, v1 in zip(v_ch, v_s)])
    elif p == 'rew_only':
      rewonly_kernel = fits[fits_names.index('rewonly')]
      trans = (-1 if transitions_AB[0] == 1 else 1)
      bp_values[p] = rewonly_kernel * np.array([x if out==0 else 0 for x, out in zip((((second_steps ^ outcomes)-0.5) * trans), outcomes)])
    elif p == 'rewonly_ch':
      rewonly_kernel = fits[fits_names.index('rewonly')]
      ss_out = np.asarray([int((ss == out) == transitions_AB[0]) for ss, out in zip(second_steps, outcomes)])
      prev_ss_out = _lag(ss_out, 1)
      bp_values[p] = [rewonly_kernel if ((c == prev) and (r1 == 1)) else 0 for c, prev, r1 in zip(choices, prev_ss_out, reward_l1)]
    elif p == 'reward_bias':
      rewonly_kernel = 1
      ss_out = np.asarray([int((ss == out) == transitions_AB[0]) for ss, out in zip(second_steps, outcomes)])
      prev_ss_out = _lag(ss_out, 1)
      bp_values[p] = [rewonly_kernel if ((c == prev) and (r1 == 1)) else 0 for c, prev, r1 in zip(choices, prev_ss_out, reward_l1)]
    elif p == 'omission_bias':
      rewonly_kernel = 1
      ss_out = np.asarray([int((ss == out) == transitions_AB[0]) for ss, out in zip(second_steps, outcomes)])
      prev_ss_out = _lag(ss_out, 1)
      bp_values[p] = [rewonly_kernel if ((c == prev) and (r1 == 0)) else 0 for c, prev, r1 in zip(choices, prev_ss_out, reward_l1)]
    elif p == 'rew_kernel_ch':
      ss_out = np.asarray([int((ss == out) == transitions_AB[0]) for ss, out in zip(second_steps, outcomes)])
      prev_ss_out = _lag(ss_out, 1)
      bp_values[p] = [1 if (c == prev) else 0 for c, prev in zip(choices, prev_ss_out)]
    elif p == 'nonrewonly_ch':
      rewonly_kernel = fits[fits_names.index('nonrewonly')]
      ss_out = np.asarray([int((ss == out) == transitions_AB[0]) for ss, out in zip(second_steps, outcomes)])
      prev_ss_out = _lag(ss_out, 1)
      bp_values[p] = [rewonly_kernel if ((c == prev) and (r1 == 0)) else 0 for c, prev, r1 in zip(choices, prev_ss_out, reward_l1)]
    elif p == 'rewonly_nonrewonly_ch':
      rewonly_kernel = fits[fits_names.index('rewonly')]
      nonrewonly_kernel = fits[fits_names.index('nonrewonly')]
      ss_out = np.asarray([int((ss == out) == transitions_AB[0]) for ss, out in zip(second_steps, outcomes)])
      prev_ss_out = _lag(ss_out, 1)
      bp_values[p] = [rewonly_kernel if ((c == prev) and (r1 == 1)) else
                      nonrewonly_kernel if ((c == prev) and (r1 == 0)) else
                      0 for c, prev, r1 in zip(choices, prev_ss_out, reward_l1)]
    elif p == 'multiperseveration':
      alphamultipersv = fits[fits_names.index('alphamultipersv')]
      multpersv_kernel = fits[fits_names.index('multpersv')]
      multipersv = np.zeros(len(choices))
      multipersv[0] = 0.5
      for i, c in enumerate(choices[:-1]):
        multipersv[i + 1] = ((1 - alphamultipersv) * multipersv[i]) + (alphamultipersv * c)
      multipersv -= 0.5
      bp_values[p] = multpersv_kernel * multipersv
    elif p == 'high_confidence':
      p_r = 0.05
      p_o_1 = np.array([[0.8, 1 - 0.8],  # Probability of observed outcome given world in state 1.
                        [1 - 0.8, 0.8]])  # Indicies:  p_o_1[second_step, outcome]
      p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.
      p_1 = np.zeros(session.n_trials + 1)
      V_p_1 = np.zeros((session.n_trials + 1, 2))
      p_1[0] = 0.5
      for i, (c, s, o) in enumerate(zip(session.trial_data['choices'], session.trial_data['second_steps'], session.trial_data['outcomes'])):  # loop over trials.

        # Bayesian update of state probabilties given observed outcome.
        p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
        # Update of state probabilities due to possibility of block reversal.
        p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

        V_p_1[i + 1, 0] = 1 - p_1[i + 1]
        V_p_1[i + 1, 1] = p_1[i + 1]

      p_ch = [val[c] for val, c in zip(V_p_1[:session.n_trials], session.trial_data['choices'])]
      bp_values[p] = p_ch

    if extra_predictors != []:
      for ep in extra_predictors:
        if p == ep[0] + 'repeat':
          bp_values[p] = np.repeat(ep[1], len(choices))
        elif p == ep[0] + 'repeat_minus':
          bp_values[p] = np.repeat(1 - ep[1], len(choices))
        elif p == ep[0]:
          bp_values[p] = np.asarray(ep[1])

    if fits_names != []:
      for fi, f in enumerate(fits_names):
        if p == f:
          bp_values[p] = np.repeat(fits[fi], len(choices))

  # Generate lagged predictors from base predictors.
  n_trials = len(choices)
  predictors = np.zeros([n_trials, n_predictors])
  predictors_future = np.zeros([n_trials, n_predictors])
  predictors_nolag= np.zeros([n_trials, len(base_predictors)])

  p_l = []
  p_l_i = 0
  for i, p in enumerate(predictors_lags):
    if '-' in p:  # Get lag from predictor name.
      lag = int(p.split('-')[1])
      bp_name = p.split('-')[0]
      predictors[lag:, i] = bp_values[bp_name][:-lag]
      if lags_future:
        predictors_future[:-lag, i] = bp_values[bp_name][lag:]
        if bp_name != p_l:
          lag = 0
          predictors_nolag[lag:,p_l_i] = bp_values[bp_name][:]
          p_l = bp_name
          p_l_i += 1

    elif single_lag:
      lag = single_lag
      bp_name = p
      predictors[lag:, i] = bp_values[bp_name][:-lag]
    else:  # Use default lag, 0 - no lag.
      lag = 0
      bp_name = p
      predictors[lag:, i] = bp_values[bp_name][:]

  if lags_future:
    return choices, predictors, predictors_future

  else:
    return (choices, predictors)

#%% optogenetics
def sum_dicts(ds):
  '''
  ds: multiple dictionaries in a list e.g. ds = [dict1_sub, dict_2_sub]
  '''
  d = {}
  for k in ds[0].keys():
    d[k] = [np.mean(np.concatenate(tuple(d[k] for d in ds)))]
  return d

def import_ICSS_4pokes(dir_folder_session):
  '''
  import intracranial self-stimulation data - version where 4 pokes were available
  '''
  experiment = di.Experiment(dir_folder_session)
  sessions_4 = experiment.get_sessions(when='2020-11-04')
  sessions_4_stim = [x for x in sessions_4 if x.variables['extiction'] == [] or x.variables['extiction'] == ['False']]
  sessions_4_extinction = [x for x in sessions_4 if x.variables['extiction'] == ['True']]

  list_pokes = ['poke_2', 'poke_3', 'poke_7', 'poke_8']
  active_pokes_subject = {x.subject_ID: x.block_data['active_poke'] for x in sessions_4_stim}

  dict_ext = {x.subject_ID: [] for x in sessions_4_extinction}
  dict_stim = {x.subject_ID: [] for x in sessions_4_stim}

  for si, session in enumerate(sessions_4_extinction):
    times_poke_active = np.asarray(
      [x.time for x in session.events_and_print if x.name == active_pokes_subject[session.subject_ID][0]])
    non_active_pokes = [x for x in list_pokes if x != active_pokes_subject[session.subject_ID][0]]
    times_poke_non_active_1 = np.asarray([x.time for x in session.events_and_print if x.name == non_active_pokes[0]])
    times_poke_non_active_2 = np.asarray([x.time for x in session.events_and_print if x.name == non_active_pokes[1]])
    times_poke_non_active_3 = np.asarray([x.time for x in session.events_and_print if x.name == non_active_pokes[2]])
    time_chunks = [0] + [session.events_and_print[-1].time / 5 * x for x in np.arange(1, 6)]
    active_counts_ext = [len(np.where(times_poke_active < time_chunks[i + 1])[0]) for i in range(len(time_chunks) - 1)]
    non_active_counts_1_ext = [len(np.where(times_poke_non_active_1 < time_chunks[i + 1])[0]) for i in
                               range(len(time_chunks) - 1)]
    non_active_counts_2_ext = [len(np.where(times_poke_non_active_2 < time_chunks[i + 1])[0]) for i in
                               range(len(time_chunks) - 1)]
    non_active_counts_3_ext = [len(np.where(times_poke_non_active_3 < time_chunks[i + 1])[0]) for i in
                               range(len(time_chunks) - 1)]

    dict_ext[session.subject_ID] = {'time_chuncks': time_chunks, 'active_counts': active_counts_ext,
                                    'non_active_counts_1': non_active_counts_1_ext,
                                    'non_active_counts_2': non_active_counts_2_ext,
                                    'non_active_counts_3': non_active_counts_3_ext}

  for si, session in enumerate(sessions_4_stim):
    times_poke_active = np.asarray(
      [x.time for x in session.events_and_print if x.name == active_pokes_subject[session.subject_ID][0]])
    non_active_pokes = [x for x in list_pokes if x != active_pokes_subject[session.subject_ID][0]]
    times_poke_non_active_1 = np.asarray([x.time for x in session.events_and_print if x.name == non_active_pokes[0]])
    times_poke_non_active_2 = np.asarray([x.time for x in session.events_and_print if x.name == non_active_pokes[1]])
    times_poke_non_active_3 = np.asarray([x.time for x in session.events_and_print if x.name == non_active_pokes[2]])
    time_chunks = [0] + [session.events_and_print[-1].time / 40 * x for x in np.arange(1, 41)]
    active_counts_stim = [len(np.where(times_poke_active < time_chunks[i + 1])[0]) for i in range(len(time_chunks) - 1)]
    non_active_counts_1_stim = [len(np.where(times_poke_non_active_1 < time_chunks[i + 1])[0]) for i in
                                range(len(time_chunks) - 1)]
    non_active_counts_2_stim = [len(np.where(times_poke_non_active_2 < time_chunks[i + 1])[0]) for i in
                                range(len(time_chunks) - 1)]
    non_active_counts_3_stim = [len(np.where(times_poke_non_active_3 < time_chunks[i + 1])[0]) for i in
                                range(len(time_chunks) - 1)]

    dict_stim[session.subject_ID] = {'time_chuncks': time_chunks, 'active_counts': active_counts_stim,
                                     'non_active_counts_1': non_active_counts_1_stim,
                                     'non_active_counts_2': non_active_counts_2_stim,
                                     'non_active_counts_3': non_active_counts_3_stim}

  return dict_stim, dict_ext

def create_dictionary_counts_2pokes(sessions):
  '''
  Create dictionary with the number of pokes to the active (stimulated) and inactive port - version of intracranial self-stimulation
  where only two ports were available
  '''
  list_pokes = ['poke_2', 'poke_3']
  active_pokes_subject = {x.subject_ID: x.block_data['active_poke'][0] for x in sessions}
  dict_ext = {x.subject_ID: [] for x in sessions}
  dict_stim = {x.subject_ID: [] for x in sessions}
  for session in sessions:
    times_poke_active = np.asarray(
      [x.time for x in session.events_and_print if x.name == active_pokes_subject[session.subject_ID]])
    non_active_pokes = [x for x in list_pokes if x != active_pokes_subject[session.subject_ID]]
    times_poke_non_active_1 = np.asarray([x.time for x in session.events_and_print if x.name == non_active_pokes[0]])
    dict_stim[session.subject_ID] = {'active_counts': [sum(times_poke_active < session.block_data['start_block_time'][-1])],
                                     'non_active_counts_1': [sum(times_poke_non_active_1 < session.block_data['start_block_time'][-1])],
                                     'non_active_counts_2': [0],
                                     'non_active_counts_3': [0]}
    dict_ext[session.subject_ID] = {'active_counts': [sum(times_poke_active >= session.block_data['start_block_time'][-1])],
                                     'non_active_counts_1': [sum(times_poke_non_active_1 >= session.block_data['start_block_time'][-1])],
                                     'non_active_counts_2': [0],
                                     'non_active_counts_3': [0]}

  return dict_stim, dict_ext

def import_all_ICSS(dir_folder_session):
  '''
  import all intracranial self-stimulation experiments
    dict_stim_4: ICSS with 4 ports
    dict_stim_5 and dict_stim_9: ICSS with two ports
    In all three sessions, only one port delivered stimulation.
    This function concatenated all poke counts across the 3 sessions
  '''
  dict_stim, dict_ext = import_ICSS_4pokes(dir_folder_session=dir_folder_session)
  dict_stim_4 = {k: {'active_counts': [dict_stim[k]['active_counts'][-1]],
                     'non_active_counts_1': [dict_stim[k]['non_active_counts_1'][-1]],
                     'non_active_counts_2': [dict_stim[k]['non_active_counts_2'][-1]],
                     'non_active_counts_3': [dict_stim[k]['non_active_counts_3'][-1]]
                     } for k in dict_stim.keys()}
  dict_ext_4 = {k: {'active_counts': [dict_ext[k]['active_counts'][-1]],
                    'non_active_counts_1': [dict_ext[k]['non_active_counts_1'][-1]],
                    'non_active_counts_2': [dict_ext[k]['non_active_counts_2'][-1]],
                    'non_active_counts_3': [dict_ext[k]['non_active_counts_3'][-1]]
                    } for k in dict_ext.keys()}

  experiment = di.Experiment(dir_folder_session)

  sessions_5 = experiment.get_sessions(when='2020-11-05')
  sessions_9 = experiment.get_sessions(when='2020-11-09')

  dict_stim_5, dict_ext_5 = create_dictionary_counts_2pokes(sessions_5)
  dict_stim_9, dict_ext_9 = create_dictionary_counts_2pokes(sessions_9)

  dict_stim_3_sessions = {sub: sum_dicts([dict_stim_4[sub], dict_stim_5[sub], dict_stim_9[sub]])
  if (sub in dict_stim_5.keys()) and (sub in dict_stim_9.keys()) else
  sum_dicts([dict_stim_4[sub], dict_stim_5[sub]])
  if sub in dict_stim_5.keys() else dict_stim_4[sub] for sub in dict_stim_4.keys()}

  return dict_stim_3_sessions

def calculate_stim_preference(dict_stim, dict_ext, subjects, percentage=True, extinction=True):
  '''
  Calculate the percentage or actual counts to active vs inactive port.
  If extinction is True, also include during the extinction stage at the end of the session
  '''
  if percentage:
    total_counts_stim = [dict_stim[sub]['active_counts'][-1] +
                    dict_stim[sub]['non_active_counts_1'][-1] +
                    dict_stim[sub]['non_active_counts_2'][-1] +
                    dict_stim[sub]['non_active_counts_3'][-1]
                    for sub in subjects]
    preference_stim = [dict_stim[sub]['active_counts'][-1]/total_counts_stim[i]*100 if total_counts_stim[i] != 0 else 0 for i,sub in enumerate(subjects)]
    if extinction:
      total_counts_ext = [dict_ext[sub]['active_counts'][-1] +
                      dict_ext[sub]['non_active_counts_1'][-1] +
                      dict_ext[sub]['non_active_counts_2'][-1] +
                      dict_ext[sub]['non_active_counts_3'][-1]
                      for sub in subjects]
      preference_ext = [dict_ext[sub]['active_counts'][-1]/total_counts_ext[i]*100 if total_counts_ext[i] != 0 else 0 for i, sub in enumerate(subjects)]
    else:
      preference_ext = []
  else:
    preference_stim = [dict_stim[sub]['active_counts'][-1] for i, sub in enumerate(subjects)]
    if extinction:
      preference_ext = [dict_ext[sub]['active_counts'][-1] for i, sub in enumerate(subjects)]
    else:
      preference_ext = []

  return preference_stim, preference_ext

def plot_poke_preference(preference_YFP_stim, preference_YFP_ext, preference_ChR2_stim, preference_ChR2_ext,
                         labels=['YFP', 'ChR2'], colors=['gold', 'blue'], percentage=True, extinction=True, figsize=(2,3),
                         scatter=True, boxplot=False):
  '''
  Plot active vs inactive port preference during the intracranial self-stimulation experiment
  '''
  x = np.arange(len(labels))
  width = 0.15
  fig, ax = plt.subplots(figsize=figsize)
  if extinction:
    rects1 = ax.bar(x - width / 2, [np.mean(preference_YFP_stim), np.mean(preference_ChR2_stim)], width,
                    yerr=[stats.sem(preference_YFP_stim), stats.sem(preference_ChR2_stim)], label='LED ON', hatch="/",
                    error_kw={'ecolor': 'k', 'capsize': 3, 'elinewidth': 2, 'markeredgewidth': 2}, color=colors,
                    linewidth=1, fill=True)
    rects2 = ax.bar(x + width / 2, [np.mean(preference_YFP_ext), np.mean(preference_ChR2_ext)], width,
                    yerr=[stats.sem(preference_YFP_ext), stats.sem(preference_ChR2_ext)], label='LED OFF (Extinction)',
                    error_kw={'ecolor': 'k', 'capsize': 3, 'elinewidth': 2, 'markeredgewidth': 2}, color=colors,
                    linewidth=1, fill=True)
  else:
    if not boxplot:
      rects1 = ax.bar(x, [np.mean(preference_YFP_stim), np.mean(preference_ChR2_stim)],
                      yerr=[stats.sem(preference_YFP_stim), stats.sem(preference_ChR2_stim)], label='LED ON',
                      error_kw={'ecolor': 'k', 'capsize': 3, 'elinewidth': 2, 'markeredgewidth': 2}, color=colors,
                      linewidth=1, fill=True, zorder = -1)
    else:
      rects1 = ax.boxplot(preference_YFP_stim, positions=[x[0]], widths = 0.6, patch_artist=True,
            boxprops=dict(facecolor=colors[0], color='black'),
                          flierprops=dict(marker='d', markersize=4, markerfacecolor='black'),
                          medianprops=dict(color='black')
            )
      rects2 = ax.boxplot(preference_ChR2_stim, positions=[x[1]], widths = 0.6, patch_artist=True,
            boxprops=dict(facecolor=colors[1], color='black'),
                          flierprops=dict(marker='d', markersize=4, markerfacecolor='black'),
                          medianprops=dict(color='black')
            )
    tstat, p_val = stats.ttest_ind(preference_YFP_stim, preference_ChR2_stim)
    print('YPF: {}'.format(preference_YFP_stim))
    print('ChR2: {}'.format(preference_ChR2_stim))
    print('dof: {}'.format(len(preference_YFP_stim) + len(preference_ChR2_stim) -2))
    print('tstat: {}'.format(tstat))
    print('pval: {}'.format(p_val))

    res = pg.ttest(preference_YFP_stim, preference_ChR2_stim, correction=False)
    print(res)
    if p_val < 0.1:
      ax.text(0.5, np.max([np.mean(preference_YFP_stim), np.mean(preference_ChR2_stim)]) +
               np.max([stats.sem(preference_YFP_stim), stats.sem(preference_ChR2_stim)]) + 0.05,
               stats_annotation([p_val])[0], ha='center', size=18)

    if scatter == True:
      x0 = np.random.normal(0, 0.14, size=len(preference_YFP_stim))
      ax.scatter(x0, preference_YFP_stim, c='dimgray', s=5, lw=0, zorder = 1)
      x0 = np.random.normal(0, 0.14, size=len(preference_ChR2_stim))
      ax.scatter(x0+1, preference_ChR2_stim, c='dimgray', s=5, lw=0, zorder = 1)

  if percentage:
    ax.set_ylabel('Preference stimulated poke (%)')
  else:
    ax.set_ylabel('# stimulated poke')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()
  plt.tight_layout()

def plot_all_ICSS(dir_folder_session, figsize=(2,3), scatter=True, boxplot=False):
  '''
  Plot all 3 last sessions with stimulations of 5 and 10 pulses
  '''
  dict_stim_3_sessions = import_all_ICSS(dir_folder_session=dir_folder_session)

  subjects_ChR2 = [64, 65, 67, 70, 72, 73, 74]
  subjects_YFP = [63, 66, 69, 71, 75]
  all_preference = [calculate_stim_preference(dict_stim_3_sessions, [], subjects, percentage=False, extinction=False)
                    for subjects in [subjects_YFP, subjects_ChR2]]
  plot_poke_preference(all_preference[0][0], all_preference[0][1], all_preference[1][0], all_preference[1][1],
                       percentage=False, extinction=False, figsize=figsize, scatter=scatter, boxplot=boxplot)

def df_import_opto_sessions_cohort(dir_folder_session, subjects_virus, sessions_to_exclude={}):
  '''
  Import behavioural sessions where optogenetic stimulation was delivered as a dataframe
  :param subjects_virus: dictionary {subject id: virus}
  :param sessions_to_exclude: dictionary {subject id: [YYYY-MM-DD,...]}
  :return: pandas dataframe with all the sessions
  '''
  experiment = di.Experiment(dir_folder_session)

  sessions_opto = experiment.get_sessions()
  sessions_opto = [s for s in sessions_opto if
                   s.subject_ID not in sessions_to_exclude.keys() or s.datetime_string.split()[0] not in
                   sessions_to_exclude[s.subject_ID]]

  stim_type = [s.trial_data['stim_type'][0] for s in sessions_opto]
  virus = [subjects_virus[s.subject_ID] for s in sessions_opto]
  subject = [s.subject_ID for s in sessions_opto]
  date = [s.datetime_string.split()[0] for s in sessions_opto]

  all_sessions = {'subject': subject,
                  'date': date,
                  'virus': virus,
                  'stim_type': stim_type,
                  'sessions': sessions_opto,
                  }

  return pd.DataFrame(data=all_sessions)

def _summary_per_subject_stim_vs_non_stim(sessions, attribute):
  '''
  Return behavioural parameter per subject for stimulated and non-stimulated trials on free-choice trials.
  For non-stimulated trials, just trials that had the same probability of being stimulated or not are taken into account
  The behavioural parameter extracted is defined by the 'attribute' variable.
  Possible attribute:
    n_trials,
    n_rewards,
    n_blocks,
    latency_initiate: median latency to initiate trial when previous trial was or notstimulated
    latency_initiate_rew: median latency to initiate trial when previous trial was or not stimulated and rewarded
    latency_initiate_unrew: median latency to initate trial when previous trial was or not stimulated and unrewarrded
    latency_initiate_rew_single: for each trial, latency to initiate trial when previous trial was or not stimulated and rewarded
    latency_initiate_unrew_single: for each trial, latency to initiate trial when previous trial was or not stimulated and unrewarded
    latency_initiate_ch_common_same_ss: median latency to initiate trial when previous trial was or not stimulated and current choice commonly leads to the previous second-step state
    latency_initiate_ch_rare_same_ss: median latency to initiate trial when previous trial was or not stimulated and current choice rarely leads to the previous second-step state
    latency_poke_5: median latency to poke the initiation port after outcome delivery when previous trial was or not stimulated
    latency_poke_5_rew: median latency to poke the initiation port after outcome delivery when previous trial was or not stimulated and rewarded
    latency_poke_5_unrew: median latency to poke the initiation port after outcome delivery when previous trial was or not stimulated and unrewarded
    latency_choice: median latency to poking the choice port when previous trial was or not stimulated
    latency_choice_rew: median latency to poking the choice port when previous trial was or not stimulated and rewarded
    latency_choice_unrew: median latency to poking the choice port when previous trial was or not stimulated and unrewarded
    latency_ss: median latency from choice to second-step (after cue offset) when previous trial was or not stimulated
    latency_ss_same: median latency from choice to second-step (after cue offset) when previous trial was or not stimulated and second-step is the same as in the prervious trial
    latency_ss_diff: median latency from choice to second-step (after cue offset) when previous trial was or not stimulated and second-step is different from the prervious trial
    latency_ss_current: median latency from choice to second-step (after cue offset) when CURRENT trial was or not stimulated
    latency_ss_poke_current: median latency from choice to poking the illuminated second-step port (even when the audittory cue is still playing) when CURRENT trial was or not stimulated
    latency_ss_current_common: median latency from choice to second-step (after cue offset) when CURRENT trial was or not stimulated, and a common transition occurred
    latency_ss_current_rare: median latency from choice to second-step (after cue offset) when CURRENT trial was or not stimulated, and a rare transition occurred
  '''

  subjects = list(set([sessions[i].subject_ID for i in range(len(sessions))]))

  stim_all = [np.hstack([x.trial_data['stim'] for x in sessions if x.subject_ID == s]) for s in subjects]
  stim_1_all = [np.hstack([_lag(x.trial_data['stim'], 1) for x in sessions if x.subject_ID == s]) for s in subjects]
  stim_2_all = [np.hstack([_lag(x.trial_data['stim'], 2) for x in sessions if x.subject_ID == s]) for s in subjects]
  stim_3_all = [np.hstack([_lag(x.trial_data['stim'], 3) for x in sessions if x.subject_ID == s]) for s in subjects]
  free_choice_all = [np.hstack([x.trial_data['free_choice'] for x in sessions if x.subject_ID == s]) for s in subjects]
  free_choice_1_all = [np.hstack([_lag(x.trial_data['free_choice'], 1) for x in sessions if x.subject_ID == s]) for s in subjects]
  outcome_all = [np.hstack([x.trial_data['outcomes'] for x in sessions if x.subject_ID == s]) for s in subjects]
  outcome_1_all = [np.hstack([_lag(x.trial_data['outcomes'],1) for x in sessions if x.subject_ID == s]) for s in subjects]
  ss_all = [np.hstack([x.trial_data['second_steps'] for x in sessions if x.subject_ID == s]) for s in subjects]
  ss_1_all = [np.hstack([_lag(x.trial_data['second_steps'],1) for x in sessions if x.subject_ID == s]) for s in subjects]
  trans_state_all = [np.hstack([x.blocks['trial_trans_state'] for x in sessions if x.subject_ID == s]) for s in subjects]
  choices_to_ssl1_all = [np.hstack([(~(_lag(x.trial_data['second_steps'],1) ^ x.blocks['trial_trans_state']).astype(bool)).astype(int)
                                for x in sessions if x.subject_ID == s]) for s in subjects]
  choice_all = [np.hstack([x.trial_data['choices'] for x in sessions if x.subject_ID == s]) for s in subjects]


  if attribute == 'n_trials':
    mean_attribute = [np.mean([x.n_trials for x in sessions if x.subject_ID == s]) for s in subjects]
  elif attribute == 'n_rewards':
    mean_attribute = [np.mean([x.rewards for x in sessions if x.subject_ID == s]) for s in subjects]
  elif attribute == 'n_blocks':
    mean_attribute = [np.mean([len(x.blocks['start_trials']) for x in sessions if x.subject_ID == s]) for s in subjects]

  elif attribute == 'latency_initiate': # next trial
    lat_all = [np.hstack([session_latencies(x, 'start')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                            and (free_choice_1_all[s][i] == True))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_initiate_rew': # next trial
    lat_all = [np.hstack([session_latencies(x, 'start')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                            and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_initiate_unrew': # next trial
    lat_all = [np.hstack([session_latencies(x, 'start')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                            and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_initiate_rew_single': # next trial
    lat_all = [np.hstack([session_latencies(x, 'start')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [[lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))]
                    for s in range(len(lat_all))]

    mean_stim = [[lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                            and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))] #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_initiate_unrew_single': # next trial
    lat_all = [np.hstack([session_latencies(x, 'start')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [[lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))]
                    for s in range(len(lat_all))]

    mean_stim = [[lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                            and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))] #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_initiate_ch_common_same_ss': # next trial
    lat_all = [np.hstack([session_latencies(x, 'start')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (choice_all[s][i] == choices_to_ssl1_all[s][i]))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                            and (free_choice_1_all[s][i] == True) and (choice_all[s][i] == choices_to_ssl1_all[s][i]))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_initiate_ch_rare_same_ss': # next trial
    lat_all = [np.hstack([session_latencies(x, 'start')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (choice_all[s][i] != choices_to_ssl1_all[s][i]))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                            and (free_choice_1_all[s][i] == True) and (choice_all[s][i] != choices_to_ssl1_all[s][i]))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_poke_5': # next trial
    lat_all = [np.hstack([session_latencies(x, 'ITI_poke_5')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                      and (free_choice_1_all[s][i] == True))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_poke_5_rew': # next trial
    lat_all = [np.hstack([session_latencies(x, 'ITI_poke_5')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                      and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_poke_5_unrew': # next trial
    lat_all = [np.hstack([session_latencies(x, 'ITI_poke_5')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                      and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_choice': # if previous stimulated
    lat_all = [np.hstack([session_latencies(x, 'choice')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_all[s][i] == True) and (free_choice_1_all[s][i] == True))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                                                              and (free_choice_all[s][i] == True)
                                                                              and (free_choice_1_all[s][i] == True))])
                    for s in range(len(lat_all))]

  elif attribute == 'latency_choice_rew': # next trial
    lat_all = [np.hstack([session_latencies(x, 'choice')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))])
                    for s in range(len(lat_all))]

    num_trials = sum([len([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))])
                    for s in range(len(lat_all))])
    print('num_nonstim: {}'.format(num_trials))

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                      and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))]) #previous trial stimulated
                 for s in range(len(lat_all))]

    num_trials = sum([len([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                      and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 1))]) #previous trial stimulated
                 for s in range(len(lat_all))])
    print('num_stim: {}'.format(num_trials))

  elif attribute == 'latency_choice_unrew': # next trial
    lat_all = [np.hstack([session_latencies(x, 'choice')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))])
                    for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                      and (free_choice_1_all[s][i] == True) and (outcome_1_all[s][i] == 0))]) #previous trial stimulated
                 for s in range(len(lat_all))]

  elif attribute == 'latency_ss':
    lat_all = [np.hstack([session_latencies(x, 'second_step')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_all[s][i] == True))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                                                              and (free_choice_all[s][i] == True))])
                    for s in range(len(lat_all))]

  elif attribute == 'latency_ss_same':
    lat_all = [np.hstack([session_latencies(x, 'second_step')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_all[s][i] == True) and (outcome_all[s][i] == outcome_1_all[s][i]))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                                                              and (free_choice_all[s][i] == True)
                                                                              and (outcome_all[s][i] == outcome_1_all[s][i]))])
                    for s in range(len(lat_all))]
  elif attribute == 'latency_ss_diff':
    lat_all = [np.hstack([session_latencies(x, 'second_step')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_1_all[s][i] == 0) and stim_2_all[s][i] == 0 and (stim_3_all[s][i] == 0)
                                and (free_choice_all[s][i] == True) and (outcome_all[s][i] != outcome_1_all[s][i]))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_1_all[s][i] == 1)
                                                                              and (free_choice_all[s][i] == True)
                                                                              and (outcome_all[s][i] != outcome_1_all[s][i]))])
                    for s in range(len(lat_all))]

  elif attribute == 'latency_ss_current':
    lat_all = [np.hstack([session_latencies(x, 'second_step')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_all[s][i] == 0) and stim_1_all[s][i] == 0 and (stim_2_all[s][i] == 0)
                                and (free_choice_all[s][i] == True))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_all[s][i] == 1) and (free_choice_all[s][i] == True))])
                    for s in range(len(lat_all))]

  elif attribute == 'latency_ss_poke_current':
    lat_all = [np.hstack([session_latencies(x, 'second_step_poke')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_all[s][i] == 0) and stim_1_all[s][i] == 0 and (stim_2_all[s][i] == 0)
                                and (free_choice_all[s][i] == True))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_all[s][i] == 1)
                                                                              and (free_choice_all[s][i] == True))])
                    for s in range(len(lat_all))]

  elif attribute == 'latency_ss_current_common':
    lat_all = [np.hstack([session_latencies(x, 'second_step')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]
    transition_common = [np.hstack([x.trial_data['transitions'].astype(bool) == x.blocks['trial_trans_state']
                                    for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_all[s][i] == 0) and stim_1_all[s][i] == 0 and (stim_2_all[s][i] == 0)
                                and (transition_common[s][i] == True) and (free_choice_all[s][i] == True))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_all[s][i] == 1) and (transition_common[s][i] == True)
                                                                              and (free_choice_all[s][i] == True))])
                    for s in range(len(lat_all))]


  elif attribute == 'latency_ss_current_rare':
    lat_all = [np.hstack([session_latencies(x, 'second_step')[:x.n_trials] for x in sessions if x.subject_ID == s]) for s in subjects]
    transition_common = [np.hstack([x.trial_data['transitions'].astype(bool) == x.blocks['trial_trans_state']
                                    for x in sessions if x.subject_ID == s]) for s in subjects]

    mean_nonstim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if
                               ((stim_all[s][i] == 0) and stim_1_all[s][i] == 0 and (stim_2_all[s][i] == 0)
                                and (transition_common[s][i] == False) and (free_choice_all[s][i] == True))])
                           for s in range(len(lat_all))]

    mean_stim = [np.median([lat_all[s][i] for i in range(len(lat_all[s])) if ((stim_all[s][i] == 1)
                                                                              and (transition_common[s][i] == False)
                                                                              and (free_choice_all[s][i] == True))])
                    for s in range(len(lat_all))]
  else:
    print('attribute incorrect')

  return  mean_nonstim, mean_stim, subjects

def plot_beh_correlates_stim_vs_nostim(sessions_group, attribute, labels_group, color_per_subject=False, title=[],
                                       scatter=False, ylabel=[], ylim=[], figsize=(5,5)):
  '''
  Plot the defined behavioural attribute between stimulated and non-stimulated trials.
  See function above for available 'attribute'
  sesions_group: list of the sessions per group. e.g. [sessions_ss_opto_ChR2, sessions_outcome_opto_ChR2]
  '''
  mean_nonstim = [[] for i in range(len(sessions_group))]
  mean_stim = [[] for i in range(len(sessions_group))]
  subjects = [[] for i in range(len(sessions_group))]
  for i, sessions in enumerate(sessions_group):
    mean_nonstim[i], mean_stim[i], subjects[i] = _summary_per_subject_stim_vs_non_stim(sessions, attribute)

  x = np.arange(len(labels_group))
  width = 0.35
  fig, ax = plt.subplots(figsize=figsize)
  rects1 = ax.bar(x - width / 2, [np.mean(x) for x in mean_nonstim], width, yerr=[stats.sem(x) for x in mean_nonstim], label='Non_stim',
                  error_kw={'ecolor': 'k', 'capsize': 3, 'elinewidth': 2, 'markeredgewidth': 2},
                  linewidth=1, fill=True, color='grey')
  rects2 = ax.bar(x + width / 2, [np.mean(x) for x in mean_stim], width, yerr=[stats.sem(x) for x in mean_stim], label='Stim',
                  error_kw={'ecolor': 'k', 'capsize': 3, 'elinewidth': 2, 'markeredgewidth': 2},
                  linewidth=1, fill=True, color='skyblue')

  if color_per_subject:
    all_subjects = np.hstack(subjects)
    colors_subjects = [[color_per_subject[sub] for sub in subjects[i]] for i in range(len(subjects))]
  else:
    colors_subjects = 'k'

  if scatter == True:
    for i in x:
      plt.scatter(np.repeat(x[i] - width / 2, len(mean_nonstim[i])), mean_nonstim[i], c='k', s=5, alpha=0.7, lw=0)
      plt.scatter(np.repeat(x[i] + width / 2, len(mean_stim[i])), mean_stim[i], c='k', s=5, alpha=0.7, lw=0)
      lines = [[l, list(zip(np.repeat(x[i] + width / 2, len(mean_stim[i])), mean_stim[i]))[il]]
               for il, l in enumerate(zip(np.repeat(x[i] - width / 2, len(mean_nonstim[i])), mean_nonstim[i]))]

      lc = LineCollection(lines, colors=colors_subjects[i])
      ax.add_collection(lc)

  if ylabel == []:
    ax.set_ylabel(attribute)
  else:
    ax.set_ylabel(ylabel)
  if ylim:
    ax.set_ylim(ylim[0], ylim[1])
  ax.set_title(title)
  ax.set_xticks(x)
  ax.set_xticklabels(labels_group, rotation=45)
  legend_stim = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

  for i, label in enumerate(labels_group):
    t, prob = stats.ttest_rel(mean_nonstim[i], mean_stim[i])
    print('{}: {}/{}'.format(label, np.mean(mean_nonstim[i]), np.mean(mean_stim[i])))
    print('t: {}, p: {}'.format(t, prob))
    res = pg.ttest(np.asarray(mean_nonstim[i]), np.asarray(mean_stim[i]), paired=True)
    print(res)

    if prob < 0.1:
      if not ylim:
        max_y = np.max([mean_nonstim, mean_stim])
        ylim = (0, max_y+50)
      plt.text(i, ylim[1]-50, stats_annotation([prob])[0], ha='center', size=18)

  if color_per_subject:
    markers = [plt.Line2D([0], [0], marker='o', color=c, ls="", mec=None, markersize=10) for c in np.hstack(colors_subjects)]
    ax.legend(markers, all_subjects, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().add_artist(legend_stim)
  plt.tight_layout()
