# -------------------------------------------------------------------------------------
# Code with functions to plot photometry data
# Marta Blanco-Pozo, 2023
# -------------------------------------------------------------------------------------

import numpy as np
from scipy import interpolate
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import warnings
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from functools import partial

from Code_final_manuscript.code import plot_behaviour as pl, parallel_processing as pp

def nan_helper(y):
  """ from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
  Helper to handle indices and logical indices of NaNs.

  Input:
      - y, 1d numpy array with possible NaNs
  Output:
      - nans, logical indices of NaNs
      - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
  Example:
     linear interpolation of NaNs
      nans, x= nan_helper(y)
      y[nans]= np.interp(x(nans), x(~nans), y[~nans])
  """

  return np.isnan(y), lambda z: z.nonzero()[0]

def get_index_time(photometry, time):
  '''
  photometry: time of photometry signal
  time: time stamp to find in the photometry signal
  return index where time is in photometry signal time
  '''
  idx = np.nanargmin(np.abs(photometry - time))
  return idx

def select_event_by_trial_type(session, event1_id, event2_id, trial_type='all', selection_type='end',
                               select_n=15, block_type='all', free_choice='all'):
  if trial_type == 'all':
    return event1_id, event2_id
  else:
    first_print_id = [i for i, x in enumerate([e.name for e in session.events_and_print]) if type(x) == list][0]
    first_init_id = np.where(np.asarray([e.name for e in session.events]) == 'init_trial')[0][0]
    if first_print_id < first_init_id:
      raise ValueError('problem aligning trials')

    start_id, end_id, _, _ = consecutive_events(session, 'init_trial', 'inter_trial_interval',
                                                ['init_trial', 'inter_trial_interval']) #id start and end of each trial

    trial_id1 = [np.where(e1 < np.asarray(end_id))[0][0] for e1 in event1_id[:min(len(end_id), len(event1_id))]
                 if np.where(e1 < np.asarray(end_id))[0] != []] #trial id first event
    trial_id2 = [np.where(e2 < np.asarray(end_id))[0][0] for e2 in event2_id[:min(len(end_id), len(event2_id))]
                 if np.where(e2 < np.asarray(end_id))[0] != []] #trial id second event

    trial_type_dict = select_trial_types_to_analyse(session, selection_type, select_n, block_type)

    if free_choice == 'all':
      trial_type_id = [x for x in trial_type_dict[trial_type]]
    elif free_choice == True:
      trial_type_id = [x for x in trial_type_dict[trial_type] if x in trial_type_dict['free_choice']]
    else:
      trial_type_id = [x for x in trial_type_dict[trial_type] if x in trial_type_dict['forced_choice']]

    sel_event_id1 = [event1_id[i] for i in range(len(trial_id1)) if trial_id1[i] in trial_type_id]
    sel_event_id2 = [event2_id[i] for i in range(len(trial_id2)) if trial_id2[i] in trial_type_id]

  return sel_event_id1, sel_event_id2

def consecutive_events(session, event1_name, event2_name, all_events_names):
  # all_events_names = ['choose_left', 'choose_right', 'poke_1', 'poke_9']
  all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                               if session.events[i].name in all_events_names])

  event1_id, event2_id, pos, pos1 = zip(*[(all_id[i], all_id[i + 1], i, i+1) for i in range(len(all_id_names) - 1)
                               if all_id_names[i] == event1_name and all_id_names[i + 1] == event2_name])

  return event1_id, event2_id, pos, pos1

def get_times_consecutive_events(session, event1, event2, possible_events, trial_type = 'all', selection_type='end',
                                 select_n=15, block_type='all', free_choice='all'):
  '''
  event1: name of first event
  event2: name of the next consecutive event
  possible_events: list of names of possible events that could happen between these two events (including
                   event1 and event2), so you just get the time stamps when event1 and event2 happen
                   consecutively.
  trial_type: trial type to analyse = 'all', 'common_trans', 'rare_trans', 'reward_trials', 'nonreward_trials',
                                      'common_trans_rew', 'rare_trans_rew', 'common_trans_nonrew', 'rare_trans_nonrew',
                                      'correct_trials', 'incorrect_trials', 'neutral_trials',
                                      'correct_common_trans_rew', 'correct_common_trans_nonrew',
                                      'correct_rare_trans_nonrew', 'correct_rare_trans_rew',
                                      'incorrect_common_trans_nonrew', 'incorrect_common_trans_rew',
                                      'incorrect_rare_trans_rew', 'incorrect_rare_trans_nonrew',
                                      'neutral_trials_common_trans_rew', 'neutral_trials_common_trans_nonrew',
                                      'neutral_trials_rare_trans_rew', 'neutral_trials_rare_trans_nonrew'.
  selection_type: 'start', 'end', 'xtr', 'all' - analyse first, last, middle, all, after block change
  select_n: number of trials to select or elimianate (depending on selection_type)
  block_type: 'all', 'neutral', 'non_neutral'
  free_choice: True or False
  return times of event1 and event2
  '''

  event1_id, event2_id, _, _ = consecutive_events(session, event1, event2, possible_events)
  sel_event1_id, sel_event2_id = select_event_by_trial_type(session, event1_id, event2_id, trial_type, selection_type,
                                                            select_n, block_type, free_choice)
  times_e1 = [session.events[sel_event1_id[i]][0] for i in range(len(sel_event1_id))]
  times_e2 = [session.events[sel_event2_id[i]][0] for i in range(len(sel_event2_id))]
  return times_e1, times_e2


def select_trial_types_to_analyse(session, selection_type='end', select_n=15, block_type='all', hemisphere='L',
                                  extra_variable=[], extra_variable_name=[]):
  if selection_type == 'mov_average_high':
    positions = session.select_trials(selection_type='all', select_n=select_n, block_type=block_type)
    positions_1 = np.where(positions == True)[0]
    positions_2 = np.where(np.asarray(session.trial_data['mov_average']) > 0.60)[0]
    positions_id = [x for x in positions_1 if x in positions_2]
  elif selection_type == 'mov_average_low':
    positions = session.select_trials(selection_type='all', select_n=select_n, block_type=block_type)
    positions_1 = np.where(positions == True)[0]
    positions_2 = np.where(np.asarray(session.trial_data['mov_average']) < 0.50)[0]
    positions_id = [x for x in positions_1 if x in positions_2]
  elif selection_type == 'reward_mov_average_high':
    positions = session.select_trials(selection_type='all', select_n=select_n, block_type=block_type)
    positions_1 = np.where(positions == True)[0]
    rew_mov_average = pl.session_mov_average_attribute(session, moving_average_variable='outcomes', tau=10)
    positions_2 = np.where(rew_mov_average > 0.70)[0]
    positions_id = [x for x in positions_1 if x in positions_2]
  elif selection_type == 'reward_mov_average_low':
    positions = session.select_trials(selection_type='all', select_n=select_n, block_type=block_type)
    positions_1 = np.where(positions == True)[0]
    rew_mov_average = pl.session_mov_average_attribute(session, moving_average_variable='outcomes', tau=10)
    positions_2 = np.where(rew_mov_average < 0.40)[0]
    positions_id = [x for x in positions_1 if x in positions_2]
  elif selection_type == 'reward_mov_average_medium':
    positions = session.select_trials(selection_type='all', select_n=select_n, block_type=block_type)
    positions_1 = np.where(positions == True)[0]
    rew_mov_average = pl.session_mov_average_attribute(session, moving_average_variable='outcomes', tau=10)
    positions_2 = np.where(np.logical_and(rew_mov_average >= 0.40, rew_mov_average <= 0.7))[0]
    positions_id = [x for x in positions_1 if x in positions_2]
  else:
    positions = session.select_trials(selection_type=selection_type, select_n=select_n, block_type=block_type)
    positions_id = np.where(positions == True)[0]
  # trials to select, e.g. eliminate 15 trials after block change

  choices = session.trial_data['choices']  # 1 - left; 0 - right
  second_steps = session.trial_data['second_steps']  # 1 - up; 0 - down
  outcomes = session.trial_data['outcomes']  # 1 - rewarded;  0 - non-rewarded
  transitions = session.trial_data['transitions']  # 1 - A transition (left-up / right-down)
                                                              # 0 - B transition (right-up / left-down)
  transition_type = session.blocks['trial_trans_state']  # 1 - block type A
                                                                    # 0 - block type B
  reward_type = session.blocks['trial_rew_state']  # 1 - reward commonly up
                                                              # 2 - neutral block
                                                              # 0 - reward commonly down
  free_choice_trials = session.trial_data['free_choice']  # 1 - free choice trial
                                                                     # 0 - forced choice trial

  reward_l1 = pl._lag(outcomes, 1)
  reward_l2 = pl._lag(outcomes, 2)

  choices_l1 = pl._lag(choices, 1)
  second_steps_l1 = pl._lag(second_steps, 1)

  free_choice = np.where(free_choice_trials == 1)[0]
  forced_choice = np.where(free_choice_trials == 0)[0]

  left_choice = np.where(choices == 1)[0]
  right_choice = np.where(choices == 0)[0]

  hemisphere_param = [1 if hemisphere == 'L' else -1][0]
  ipsi_contra_choice = np.asarray([0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param for c in (choices * 1)])
  ipsi_choice = np.where(ipsi_contra_choice == 0.5)[0]
  contra_choice = np.where(ipsi_contra_choice == -0.5)[0]

  up_state = np.where(second_steps == 1)[0]
  down_state = np.where(second_steps == 0)[0]

  common_trans = np.where((transitions == transition_type))[0]
  rare_trans = np.where(~(transitions == transition_type))[0]

  reward_trials = np.where(outcomes == 1)[0]
  nonreward_trials = np.where(outcomes == 0)[0]

  reward_trials_l1 = np.where(reward_l1 == 1)[0]
  nonreward_trials_l1 = np.where(reward_l1 == 0)[0]
  reward_trials_l2 = np.where(reward_l2 == 1)[0]
  nonreward_trials_l2 = np.where(reward_l2 == 0)[0]

  left_rew = np.array([x for x in left_choice if x in reward_trials])
  left_nonrew = np.array([x for x in left_choice if x in nonreward_trials])
  right_rew = np.array([x for x in right_choice if x in reward_trials])
  right_nonrew = np.array([x for x in right_choice if x in nonreward_trials])

  up_rew = np.array([x for x in up_state if x in reward_trials])
  up_nonrew = np.array([x for x in up_state if x in nonreward_trials])
  down_rew = np.array([x for x in down_state if x in reward_trials])
  down_nonrew = np.array([x for x in down_state if x in nonreward_trials])

  common_trans_rew = np.array([x for x in common_trans if x in reward_trials])
  rare_trans_rew   = np.array([x for x in rare_trans   if x in reward_trials])
  common_trans_nonrew = np.array([x for x in common_trans if x in nonreward_trials])
  rare_trans_nonrew   = np.array([x for x in rare_trans   if x in nonreward_trials])

  correct = np.where(((~choices.astype(bool)).astype(int) == transition_type ^ reward_type) == True)[0]
  incorrect = np.where(((~choices.astype(bool)).astype(int) == transition_type ^ reward_type) == False)[0]
  neutral_trials = np.where(reward_type == 2)[0]
  correct_trials = np.array([x for x in correct if x not in neutral_trials])
  incorrect_trials = np.array([x for x in incorrect if x not in neutral_trials])

  good_second_step = np.where(second_steps == reward_type)[0]
  bad_second_step = np.where(second_steps != reward_type)[0]
  good_second_step_trials = np.array([x for x in good_second_step if x not in neutral_trials])
  bad_second_step_trials = np.array([x for x in bad_second_step if x not in neutral_trials])

  good_second_step_rew = np.array([x for x in good_second_step_trials if x in reward_trials])
  good_second_step_nonrew = np.array([x for x in good_second_step_trials if x in nonreward_trials])
  bad_second_step_rew = np.array([x for x in bad_second_step_trials if x in reward_trials])
  bad_second_step_nonrew = np.array([x for x in bad_second_step_trials if x in nonreward_trials])

  correct_common_trans = np.array([x for x in correct_trials if x in common_trans])
  correct_rare_trans = np.array([x for x in correct_trials if x in rare_trans])
  incorrect_common_trans = np.array([x for x in incorrect_trials if x in common_trans])
  incorrect_rare_trans = np.array([x for x in incorrect_trials if x in rare_trans])
  neutral_common_trans = np.array([x for x in neutral_trials if x in common_trans])
  neutral_rare_trans = np.array([x for x in neutral_trials if x in rare_trans])

  correct_common_trans_rew = np.array([x for x in correct_trials if x in common_trans_rew])
  correct_common_trans_nonrew = np.array([x for x in correct_trials if x in common_trans_nonrew])
  correct_rare_trans_nonrew = np.array([x for x in correct_trials if x in rare_trans_nonrew])
  correct_rare_trans_rew = np.array([x for x in correct_trials if x in rare_trans_rew])

  incorrect_common_trans_nonrew = np.array([x for x in incorrect_trials if x in common_trans_nonrew])
  incorrect_common_trans_rew = np.array([x for x in incorrect_trials if x in common_trans_rew])
  incorrect_rare_trans_rew = np.array([x for x in incorrect_trials if x in rare_trans_rew])
  incorrect_rare_trans_nonrew = np.array([x for x in incorrect_trials if x in rare_trans_nonrew])

  neutral_trials_common_trans_rew = np.array([x for x in neutral_trials if x in common_trans_rew])
  neutral_trials_common_trans_nonrew = np.array([x for x in neutral_trials if x in common_trans_nonrew])
  neutral_trials_rare_trans_rew = np.array([x for x in neutral_trials if x in rare_trans_rew])
  neutral_trials_rare_trans_nonrew = np.array([x for x in neutral_trials if x in rare_trans_nonrew])

  correct_rew = np.array([x for x in correct_trials if x in reward_trials])
  correct_nonrew = np.array([x for x in correct_trials if x in nonreward_trials])
  incorrect_rew = np.array([x for x in incorrect_trials if x in reward_trials])
  incorrect_nonrew = np.array([x for x in incorrect_trials if x in nonreward_trials])

  latency_ITI = pl.session_latencies(session, 'ITI')
  latency_ITI = pl._lag(latency_ITI, 1)
  short_ITI = np.where(latency_ITI < 2500)[0]
  medium_ITI = np.where(latency_ITI == 3000)[0]
  long_ITI = np.where(latency_ITI >= 3500)[0]

  same_ch = (choices == choices_l1) * 1
  choices_to_ssl1 = (~(second_steps_l1 ^ transitions).astype(bool)).astype(int)
  same_ch_to_ssl1 = (choices_to_ssl1 == choices) * 1

  same_ss = (second_steps == second_steps_l1) * 1
  same_ss_rew = np.asarray([1 if (ss == 1 and r1 == 1) else 0 for ss, r1 in zip(same_ss, reward_l1)])
  same_ss_nonrew = np.asarray([1 if (ss == 1 and r1 == 0) else 0 for ss, r1 in zip(same_ss, reward_l1)])
  diff_ss_rew = np.asarray([1 if (ss == 0 and r1 == 1) else 0 for ss, r1 in zip(same_ss, reward_l1)])
  diff_ss_nonrew = np.asarray([1 if (ss == 0 and r1 == 0) else 0 for ss, r1 in zip(same_ss, reward_l1)])
  same_ss_rew_trials = np.where(same_ss_rew == 1)[0]
  same_ss_nonrew_trials = np.where(same_ss_nonrew == 1)[0]
  diff_ss_rew_trials = np.where(diff_ss_rew == 1)[0]
  diff_ss_nonrew_trials = np.where(diff_ss_nonrew == 1)[0]

  same_ch_trials = np.where(same_ch == 1)[0]
  diff_ch_trials = np.where(same_ch == 0)[0]
  same_ch_to_ssl1_trials = np.where(same_ch_to_ssl1 == 1)[0]
  diff_ch_to_ssl1_trials = np.where(same_ch_to_ssl1 == 0)[0]

  same_ch_rew1_rew = np.array([x for x in same_ch_trials if ((x in reward_trials_l1) and (x in reward_trials))])
  same_ch_nonrew1_rew = np.array([x for x in same_ch_trials if ((x in nonreward_trials_l1) and (x in reward_trials))])
  same_ch_rew1_nonrew = np.array([x for x in same_ch_trials if ((x in reward_trials_l1) and (x in nonreward_trials))])
  same_ch_nonrew1_nonrew = np.array([x for x in same_ch_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials))])

  diff_ch_rew1_rew = np.array([x for x in diff_ch_trials if ((x in reward_trials_l1) and (x in reward_trials))])
  diff_ch_nonrew1_rew = np.array([x for x in diff_ch_trials if ((x in nonreward_trials_l1) and (x in reward_trials))])
  diff_ch_rew1_nonrew = np.array([x for x in diff_ch_trials if ((x in reward_trials_l1) and (x in nonreward_trials))])
  diff_ch_nonrew1_nonrew = np.array([x for x in diff_ch_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials))])

  same_ch_to_ssl1_rew1_rew = np.array([x for x in same_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in reward_trials))])
  same_ch_to_ssl1_nonrew1_rew = np.array([x for x in same_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in reward_trials))])
  same_ch_to_ssl1_rew1_nonrew = np.array([x for x in same_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in nonreward_trials))])
  same_ch_to_ssl1_nonrew1_nonrew = np.array([x for x in same_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials))])

  diff_ch_to_ssl1_rew1_rew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in reward_trials))])
  diff_ch_to_ssl1_nonrew1_rew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in reward_trials))])
  diff_ch_to_ssl1_rew1_nonrew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in nonreward_trials))])
  diff_ch_to_ssl1_nonrew1_nonrew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials))])

  same_ch_to_ssl1_common_rew1_rew = np.array([x for x in same_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in reward_trials) and (x in common_trans))])
  same_ch_to_ssl1_common_nonrew1_rew = np.array([x for x in same_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in reward_trials) and (x in common_trans))])
  same_ch_to_ssl1_common_rew1_nonrew = np.array([x for x in same_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in nonreward_trials) and (x in common_trans))])
  same_ch_to_ssl1_common_nonrew1_nonrew = np.array([x for x in same_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials) and (x in common_trans))])

  diff_ch_to_ssl1_common_rew1_rew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in reward_trials) and (x in common_trans))])
  diff_ch_to_ssl1_common_nonrew1_rew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in reward_trials) and (x in common_trans))])
  diff_ch_to_ssl1_common_rew1_nonrew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in nonreward_trials) and (x in common_trans))])
  diff_ch_to_ssl1_common_nonrew1_nonrew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials) and (x in common_trans))])

  same_ch_to_ssl1_rare_rew1_rew = np.array([x for x in same_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in reward_trials) and (x in rare_trans))])
  same_ch_to_ssl1_rare_nonrew1_rew = np.array([x for x in same_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in reward_trials) and (x in rare_trans))])
  same_ch_to_ssl1_rare_rew1_nonrew = np.array([x for x in same_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in nonreward_trials) and (x in rare_trans))])
  same_ch_to_ssl1_rare_nonrew1_nonrew = np.array([x for x in same_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials) and (x in rare_trans))])

  diff_ch_to_ssl1_rare_rew1_rew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in reward_trials) and (x in rare_trans))])
  diff_ch_to_ssl1_rare_nonrew1_rew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in reward_trials) and (x in rare_trans))])
  diff_ch_to_ssl1_rare_rew1_nonrew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in reward_trials_l1) and (x in nonreward_trials) and (x in rare_trans))])
  diff_ch_to_ssl1_rare_nonrew1_nonrew = np.array([x for x in diff_ch_to_ssl1_trials if ((x in nonreward_trials_l1) and (x in nonreward_trials) and (x in rare_trans))])

  if 'opposite_direction_choice' in extra_variable_name:
    idx = extra_variable_name.index('opposite_direction_choice')
    opposite_side = extra_variable[idx]
    direct_choice = np.where(np.asarray(opposite_side) == 0)[0]
    change_choice = np.where(np.asarray(opposite_side) == 1)[0]

    ipsi_direct = np.array([x for x in direct_choice if x in ipsi_choice])
    ipsi_change = np.array([x for x in change_choice if x in ipsi_choice])
    contra_direct = np.array([x for x in direct_choice if x in contra_choice])
    contra_change = np.array([x for x in change_choice if x in contra_choice])
  else:
    direct_choice = []
    change_choice = []
    ipsi_direct = []
    ipsi_change = []
    contra_direct = []
    contra_change = []


  return {'all': [x for x in positions_id],
          'free_choice': [x for x in free_choice if x in positions_id],
          'forced_choice': [x for x in forced_choice if x in positions_id],
          'left_choice': [x for x in left_choice if x in positions_id],
          'right_choice': [x for x in right_choice if x in positions_id],
          'ipsi_choice': [x for x in ipsi_choice if x in positions_id],
          'contra_choice': [x for x in contra_choice if x in positions_id],
          'up_state': [x for x in up_state if x in positions_id],
          'down_state': [x for x in down_state if x in positions_id],
          'common_trans': [x for x in common_trans if x in positions_id],
          'rare_trans': [x for x in rare_trans if x in positions_id],
          'reward_trials': [x for x in reward_trials if x in positions_id],
          'nonreward_trials': [x for x in nonreward_trials if x in positions_id],
          'reward_trials_l1': [x for x in reward_trials_l1 if x in positions_id],
          'nonreward_trials_l1': [x for x in nonreward_trials_l1 if x in positions_id],
          'reward_trials_l2': [x for x in reward_trials_l2 if x in positions_id],
          'nonreward_trials_l2': [x for x in nonreward_trials_l2 if x in positions_id],
          'left_rew': [x for x in left_rew if x in positions_id],
          'left_nonrew': [x for x in left_nonrew if x in positions_id],
          'right_rew': [x for x in right_rew if x in positions_id],
          'right_nonrew': [x for x in right_nonrew if x in positions_id],
          'up_rew': [x for x in up_rew if x in positions_id],
          'up_nonrew': [x for x in up_nonrew if x in positions_id],
          'down_rew': [x for x in down_rew if x in positions_id],
          'down_nonrew': [x for x in down_nonrew if x in positions_id],
          'common_trans_rew': [x for x in common_trans_rew if x in positions_id],
          'rare_trans_rew': [x for x in rare_trans_rew if x in positions_id],
          'common_trans_nonrew': [x for x in common_trans_nonrew if x in positions_id],
          'rare_trans_nonrew': [x for x in rare_trans_nonrew if x in positions_id],
          'correct_trials': [x for x in correct_trials if x in positions_id],
          'incorrect_trials': [x for x in incorrect_trials if x in positions_id],
          'neutral_trials': [x for x in neutral_trials if x in positions_id],
          'good_second_step_trials': [x for x in good_second_step_trials if x in positions_id],
          'bad_second_step_trials': [x for x in bad_second_step_trials if x in positions_id],
          'good_second_step_rew': [x for x in good_second_step_rew if x in positions_id],
          'good_second_step_nonrew': [x for x in good_second_step_nonrew if x in positions_id],
          'bad_second_step_rew': [x for x in bad_second_step_rew if x in positions_id],
          'bad_second_step_nonrew': [x for x in bad_second_step_nonrew if x in positions_id],
          'correct_common_trans': [x for x in correct_common_trans if x in positions_id],
          'correct_rare_trans': [x for x in correct_rare_trans if x in positions_id],
          'incorrect_common_trans': [x for x in incorrect_common_trans if x in positions_id],
          'incorrect_rare_trans': [x for x in incorrect_rare_trans if x in positions_id],
          'neutral_common_trans': [x for x in neutral_common_trans if x in positions_id],
          'neutral_rare_trans': [x for x in neutral_rare_trans if x in positions_id],
          'correct_common_trans_rew': [x for x in correct_common_trans_rew if x in positions_id],
          'correct_common_trans_nonrew': [x for x in correct_common_trans_nonrew if x in positions_id],
          'correct_rare_trans_nonrew': [x for x in correct_rare_trans_nonrew if x in positions_id],
          'correct_rare_trans_rew': [x for x in correct_rare_trans_rew if x in positions_id],
          'incorrect_common_trans_nonrew': [x for x in incorrect_common_trans_nonrew if x in positions_id],
          'incorrect_common_trans_rew': [x for x in incorrect_common_trans_rew if x in positions_id],
          'incorrect_rare_trans_rew': [x for x in incorrect_rare_trans_rew if x in positions_id],
          'incorrect_rare_trans_nonrew': [x for x in incorrect_rare_trans_nonrew if x in positions_id],
          'neutral_trials_common_trans_rew': [x for x in neutral_trials_common_trans_rew if x in positions_id],
          'neutral_trials_common_trans_nonrew': [x for x in neutral_trials_common_trans_nonrew if x in positions_id],
          'neutral_trials_rare_trans_rew': [x for x in neutral_trials_rare_trans_rew if x in positions_id],
          'neutral_trials_rare_trans_nonrew': [x for x in neutral_trials_rare_trans_nonrew if x in positions_id],
          'correct_rew': [x for x in correct_rew if x in positions_id],
          'correct_nonrew': [x for x in correct_nonrew if x in positions_id],
          'incorrect_rew': [x for x in incorrect_rew if x in positions_id],
          'incorrect_nonrew': [x for x in incorrect_nonrew if x in positions_id],
          'short_ITI': [x for x in short_ITI if x in positions_id],
          'medium_ITI': [x for x in medium_ITI if x in positions_id],
          'long_ITI': [x for x in long_ITI if x in positions_id],
          'direct_choice': [x for x in direct_choice if x in positions_id],
          'change_choice': [x for x in change_choice if x in positions_id],
          'ipsi_direct': [x for x in ipsi_direct if x in positions_id],
          'ipsi_change': [x for x in ipsi_change if x in positions_id],
          'contra_direct': [x for x in contra_direct if x in positions_id],
          'contra_change': [x for x in contra_change if x in positions_id],
          'same_ch_rew1_rew': [x for x in same_ch_rew1_rew if x in positions_id],
          'same_ch_nonrew1_rew': [x for x in same_ch_nonrew1_rew if x in positions_id],
          'same_ch_rew1_nonrew': [x for x in same_ch_rew1_nonrew if x in positions_id],
          'same_ch_nonrew1_nonrew': [x for x in same_ch_nonrew1_nonrew if x in positions_id],
          'diff_ch_rew1_rew': [x for x in diff_ch_rew1_rew if x in positions_id],
          'diff_ch_nonrew1_rew': [x for x in diff_ch_nonrew1_rew if x in positions_id],
          'diff_ch_rew1_nonrew': [x for x in diff_ch_rew1_nonrew if x in positions_id],
          'diff_ch_nonrew1_nonrew': [x for x in diff_ch_nonrew1_nonrew if x in positions_id],
          'same_ch_to_ssl1_rew1_rew': [x for x in same_ch_to_ssl1_rew1_rew if x in positions_id],
          'same_ch_to_ssl1_nonrew1_rew': [x for x in same_ch_to_ssl1_nonrew1_rew if x in positions_id],
          'same_ch_to_ssl1_rew1_nonrew': [x for x in same_ch_to_ssl1_rew1_nonrew if x in positions_id],
          'same_ch_to_ssl1_nonrew1_nonrew': [x for x in same_ch_to_ssl1_nonrew1_nonrew if x in positions_id],
          'diff_ch_to_ssl1_rew1_rew': [x for x in diff_ch_to_ssl1_rew1_rew if x in positions_id],
          'diff_ch_to_ssl1_nonrew1_rew': [x for x in diff_ch_to_ssl1_nonrew1_rew if x in positions_id],
          'diff_ch_to_ssl1_rew1_nonrew': [x for x in diff_ch_to_ssl1_rew1_nonrew if x in positions_id],
          'diff_ch_to_ssl1_nonrew1_nonrew': [x for x in diff_ch_to_ssl1_nonrew1_nonrew if x in positions_id],
          'same_ch_to_ssl1_common_rew1_rew': [x for x in same_ch_to_ssl1_common_rew1_rew if x in positions_id],
          'same_ch_to_ssl1_common_nonrew1_rew': [x for x in same_ch_to_ssl1_common_nonrew1_rew if x in positions_id],
          'same_ch_to_ssl1_common_rew1_nonrew': [x for x in same_ch_to_ssl1_common_rew1_nonrew if x in positions_id],
          'same_ch_to_ssl1_common_nonrew1_nonrew': [x for x in same_ch_to_ssl1_common_nonrew1_nonrew if x in positions_id],
          'diff_ch_to_ssl1_common_rew1_rew': [x for x in diff_ch_to_ssl1_common_rew1_rew if x in positions_id],
          'diff_ch_to_ssl1_common_nonrew1_rew': [x for x in diff_ch_to_ssl1_common_nonrew1_rew if x in positions_id],
          'diff_ch_to_ssl1_common_rew1_nonrew': [x for x in diff_ch_to_ssl1_common_rew1_nonrew if x in positions_id],
          'diff_ch_to_ssl1_common_nonrew1_nonrew': [x for x in diff_ch_to_ssl1_common_nonrew1_nonrew if x in positions_id],
          'same_ch_to_ssl1_rare_rew1_rew': [x for x in same_ch_to_ssl1_rare_rew1_rew if x in positions_id],
          'same_ch_to_ssl1_rare_nonrew1_rew': [x for x in same_ch_to_ssl1_rare_nonrew1_rew if x in positions_id],
          'same_ch_to_ssl1_rare_rew1_nonrew': [x for x in same_ch_to_ssl1_rare_rew1_nonrew if x in positions_id],
          'same_ch_to_ssl1_rare_nonrew1_nonrew': [x for x in same_ch_to_ssl1_rare_nonrew1_nonrew if x in positions_id],
          'diff_ch_to_ssl1_rare_rew1_rew': [x for x in diff_ch_to_ssl1_rare_rew1_rew if x in positions_id],
          'diff_ch_to_ssl1_rare_nonrew1_rew': [x for x in diff_ch_to_ssl1_rare_nonrew1_rew if x in positions_id],
          'diff_ch_to_ssl1_rare_rew1_nonrew': [x for x in diff_ch_to_ssl1_rare_rew1_nonrew if x in positions_id],
          'diff_ch_to_ssl1_rare_nonrew1_nonrew': [x for x in diff_ch_to_ssl1_rare_nonrew1_nonrew if x in positions_id],
          'same_ss_rew_trials': [x for x in same_ss_rew_trials if x in positions_id],
          'same_ss_nonrew_trials': [x for x in same_ss_nonrew_trials if x in positions_id],
          'diff_ss_rew_trials': [x for x in diff_ss_rew_trials if x in positions_id],
          'diff_ss_nonrew_trials': [x for x in diff_ss_nonrew_trials if x in positions_id],
          }

#%% Time-warping photometry signal

def get_dict_index_session_events(session, event1, event2, possible_events, join_events, trial_type, **kwargs):
  '''
  return index in session.events where the events happened
  '''

  selection_type = kwargs.pop('selection_type', 'end')
  select_n = kwargs.pop('select_n', 15)
  block_type = kwargs.pop('block_type', 'all')

  dict_events_id = {}
  dict_events_id['init_trial'], dict_events_id['inter_trial_interval'], _, _ = consecutive_events(
      session, 'init_trial', 'inter_trial_interval', ['init_trial', 'inter_trial_interval'])
  for i in range(len(event1)):
    temp_event1_id, temp_event2_id, _, _ = consecutive_events(session, event1[i], event2[i], possible_events[i])
    dict_events_id[event1[i]] = [x for x in temp_event1_id if
                                 dict_events_id['init_trial'][0] <= x <= dict_events_id['inter_trial_interval'][-1]]
    dict_events_id[event2[i]] = [x for x in temp_event2_id if
                                 dict_events_id['init_trial'][0] <= x <= dict_events_id['inter_trial_interval'][-1]]

  for je in join_events:
    dict_events_id[je] = []
    for x in join_events[je]:
      dict_events_id[je] += dict_events_id[x]
    dict_events_id[je].sort()


  dict_events_id = {key: dict_events_id[key] for key in ['init_trial', 'inter_trial_interval', 'choice_state',
                                                         'choice', 'second_step_state', 'second_step_choice',
                                                         'reward_consumption']}

  #check if each trial has been separated correctly
  if all([len(dict_events_id[x]) == len(session.trial_data['choices']) for x in dict_events_id.keys()]) == False:
    print('problem - {}'.format(session.file_name))

  trial_type_dict = select_trial_types_to_analyse(session, selection_type, select_n, block_type)
  dict_events_id = {key: [dict_events_id[key][y] for y in range(len(dict_events_id[key]))
                          if y in trial_type_dict[trial_type]] for key in dict_events_id.keys()}

  return dict_events_id

def get_dict_event_times(session, dict_events_id):
  '''
  get times when events happened
  '''

  dict_events_times = {}
  for x in dict_events_id:
    dict_events_times[x] = [session.events[dict_events_id[x][i]][0] for i in range(len(dict_events_id[x]))]
  return dict_events_times

def get_index_photometry_events(sample_times_pyc, dict_events_times, trial_time_point_align):
  '''
  get index in the photometry signal when events happened
  '''

  dict_pho_events_id = {}
  for x in trial_time_point_align:
    dict_pho_events_id[x] = [get_index_time(
      sample_times_pyc, dict_events_times[trial_time_point_align[x][0]][i] + trial_time_point_align[x][1])
      for i in range(len(dict_events_times[trial_time_point_align[x][0]]))]

  return dict_pho_events_id

def get_median_time_and_median_len_by_time_point(dict_events_times, dict_pho_events_id, trial_time_point_align):
  '''
  return the median amount of time and median number of timepoints between events
  '''
  dict_len_time = {x: [] for x in trial_time_point_align}
  dict_len_id = {x: [] for x in trial_time_point_align}
  for et in range(len(dict_events_times)):
    for x in range(len(trial_time_point_align)-1):
      times_e1 = [dict_events_times[et][trial_time_point_align[x][0]][i] + trial_time_point_align[x][1]
                  for i in range(len(dict_events_times[et][trial_time_point_align[x][0]]))]
      times_e2 = [dict_events_times[et][trial_time_point_align[x+1][0]][i] + trial_time_point_align[x+1][1]
                  for i in range(len(dict_events_times[et][trial_time_point_align[x+1][0]]))]
      event1_id = dict_pho_events_id[et][x]
      event2_id = dict_pho_events_id[et][x+1]

      dict_len_time[x].append([e2 - e1 for e1, e2 in zip(times_e1, times_e2)])
      dict_len_id[x].append([e2 - e1 for e1, e2 in zip(event1_id, event2_id)])

  dict_median_time = {x: [] for x in trial_time_point_align}
  dict_median_len = {x: [] for x in trial_time_point_align}
  for x in range(len(trial_time_point_align)-1):
    dict_median_time[x] = int(np.median(np.hstack(dict_len_time[x])))
    dict_median_len[x] = int(np.median(np.hstack(dict_len_id[x])))

  return dict_median_time, dict_median_len

def scale_time_events(median_time, median_len, corrected_signal, pho_id1_i, pho_id2_i):
  '''
  get median time length and median length of time points and scale all trials to a common time
  '''
  if (corrected_signal[pho_id1_i:pho_id2_i].size <= 1):
    # fill with the same value// fill with nan
    x = np.linspace(0, median_time, median_len)
    y_stretch = [np.nan] * len(x)
  else:
    x = np.linspace(0, median_time, median_len)
    y_interp = interpolate.interp1d(np.arange(corrected_signal[pho_id1_i:pho_id2_i].size),
                                    corrected_signal[pho_id1_i:pho_id2_i], fill_value='extrapolate')
    y_stretch = y_interp(np.linspace(0, corrected_signal[pho_id1_i:pho_id2_i].size - 1, x.size))
  return x, y_stretch

def split_data_per_trial(sessions, all_sample_times_pyc, all_corrected_signal, trial_type, time_start, time_end, **kwargs):
  '''
  Split data between consecutive events, between element in event1 and element in event2 in the same position.
  return:
    t_scale: list of the times between two consecutive events
    pho_scale: dictionary with the photometry data between two consecutive events across trials
    dict_median_time: dictionary with the median time between two consecutive events
    dict_median_len: dictionaty with the median numbere of timepoits between two consecutive events
  '''
  selection_type = kwargs.pop('selection_type', 'end')
  select_n = kwargs.pop('select_n', 15)
  block_type = kwargs.pop('block_type', 'all')

  event1 = ['choice_state', 'choice_state', 'choice_state', 'up_state', 'down_state']
  event2 = ['choose_left', 'choose_right', 'reward_consumption', 'choose_up', 'choose_down']
  possible_events = [['choice_state', 'choose_left', 'choose_right'],
                     ['choice_state', 'choose_left', 'choose_right'],
                     ['choice_state', 'reward_consumption'],
                     ['up_state', 'choose_up'],
                     ['down_state', 'choose_down']]

  join_events = {'choice': ['choose_left', 'choose_right'],
                 'second_step_state': ['up_state', 'down_state'],
                 'second_step_choice': ['choose_up', 'choose_down']}

  trial_time_point_align = {0: ['choice_state', time_start],
                            1: ['choice_state', 0],
                            2: ['choice', 0],
                            3: ['second_step_state', 0],
                            4: ['second_step_choice', 0],
                            5: ['second_step_choice', time_end]}

  dict_events_id = []
  dict_events_times = []
  dict_pho_events_id = []
  for session, sample_times_pyc in zip(sessions, all_sample_times_pyc):
    print(session.subject_ID, session.datetime_string)
    dict_events_id.append(get_dict_index_session_events(session, event1, event2, possible_events, join_events,
                                                        trial_type, selection_type=selection_type, select_n=select_n,
                                                        block_type=block_type))
    dict_events_times.append(get_dict_event_times(session, dict_events_id[-1]))
    dict_pho_events_id.append(
      get_index_photometry_events(sample_times_pyc, dict_events_times[-1], trial_time_point_align))

  print('computing median time per events sections')
  dict_median_time, dict_median_len = get_median_time_and_median_len_by_time_point(
      dict_events_times, dict_pho_events_id, trial_time_point_align)

  t_scale = {x: [] for x in trial_time_point_align}
  pho_scale = {x: [] for x in trial_time_point_align}
  for x in range(len(trial_time_point_align)-1):
    for s in range(len(sessions)):
      temp_t_scale, temp_pho_scale = zip(*[scale_time_events(
          dict_median_time[x], dict_median_len[x], all_corrected_signal[s], p1, p2)
          for p1, p2 in zip(dict_pho_events_id[s][x], dict_pho_events_id[s][x+1])])
      t_scale[x].append(np.asarray(temp_t_scale))
      pho_scale[x].append(np.asarray(temp_pho_scale))
      print('scaled session {}'.format(s))

  if len(t_scale[0]) > 1:  # check if there are more one session to check for correct scaling across sessions
    for x in range(len(t_scale) - 1):
      for i in range(len(t_scale[x])):
        if np.asarray([e == t_scale[x][i][1] for e in np.asarray(t_scale[x][1])]).all() != True:
          raise ValueError('Error in time scaling')

  t_scale = [t_scale[a][0][1] for a in range(len(trial_time_point_align)-1)]

  return t_scale, pho_scale, dict_median_time, dict_median_len

def join_trial_scaling_t_pho_scale(t_scale, pho_scale):
  '''
  return t_scale and pho_scale from the function above concatenated per each trial, it also returns the position at
  which events were concatenated
  '''
  v_line = []
  for i in range(1, len(t_scale)):
    t_scale[i] = [x + t_scale[i - 1][-1] for x in t_scale[i]]
    v_line.append(t_scale[i - 1][-1])
  t_scale_whole = np.hstack(t_scale)
  pho_scale_together = pho_scale[0]
  for i in range(1,len(pho_scale)-1):
    pho_scale_together = [np.hstack((x, y)) for x,y in zip(pho_scale_together, pho_scale[i])]

  return t_scale_whole, pho_scale_together, v_line

def get_scaled_photometry_per_trial(sessions, all_sample_times_pyc, all_corrected_signal, time_start, time_end):
  '''
  return time-warped photometry data
    t_scale_whole: array with the times of each timepoint of the time-warped trials
    pho_scale_together: list of arrays with the photometry data per session. Array shape: (trial, timepoint)
    v_line: list with the position where events were aligned
    dict_median_time: dictionary with the median time between the aligned events
    dict_median_len: dictionary with the median number of timepoints between the aligned events
  '''
  t_scale, pho_scale, dict_median_time, dict_median_len = split_data_per_trial(sessions, all_sample_times_pyc,
                                                                               all_corrected_signal, trial_type='all',
                                                                               time_start=time_start, time_end=time_end,
                                                                               selection_type='all', select_n=15,
                                                                               block_type='all')

  t_scale_whole, pho_scale_together, v_line = join_trial_scaling_t_pho_scale(t_scale, pho_scale)
  return t_scale_whole, pho_scale_together, v_line, dict_median_time, dict_median_len

# %% Structure data
def select_sessions_to_plot(sessions, all_photo_data, pho_scale_together, mouse_analyse, region_analyse,
                            hemisphere_analyse, days_analyse, return_idx=False, exclude=[]):
  '''
  return the index of the sessions selected
  exclude: [[subject_ID, region, hemisphere, day_str],[...]]
  '''

  mouse_select = [int(''.join(x for x in all_photo_data[i]['subject_ID'] if x.isdigit())) in mouse_analyse for i in range(len(all_photo_data))]
  idx_select = list(np.where(mouse_select)[0])
  if region_analyse:
    region_select = [all_photo_data[i]['region'] in region_analyse for i in range(len(all_photo_data))]
    idx_select = list(set(idx_select) & set(np.where(region_select)[0]))
  if hemisphere_analyse:
    hemisphere_select = [all_photo_data[i]['hemisphere'] in hemisphere_analyse for i in range(len(all_photo_data))]
    idx_select = list(set(idx_select) & set(np.where(hemisphere_select)[0]))
  if days_analyse:
    days_select = [all_photo_data[i]['datetime_str'].split()[0] in days_analyse for i in range(len(all_photo_data))]
    idx_select = list(set(idx_select) & set(np.where(days_select)[0]))
  if exclude:
    idx_exclude = list(np.hstack([[i for i in range(len(all_photo_data)) if
                                   (all_photo_data[i]['subject_ID'] == ex[0]) and (
                                             all_photo_data[i]['region'] == ex[1]) and (
                                             all_photo_data[i]['hemisphere'] == ex[2]) and (
                                             all_photo_data[i]['datetime_str'].split()[0] == ex[3])] for ex in
                                  exclude]))
    idx_select = [ix for ix in idx_select if ix not in idx_exclude]

  if return_idx == False:
    pho_scale_together_select = [pho_scale_together[i] for i in idx_select]
    all_photo_data_select = [all_photo_data[i] for i in idx_select]
    sessions_select = [sessions[i] for i in idx_select]
    return sessions_select, all_photo_data_select, pho_scale_together_select

  else:
    return idx_select

def dict_selected_pho_variables(dict_name, sessions_together, all_photo_data_info, pho_scale_together, all_corrected_signal_together,
                                all_sample_times_pyc, mice, regions, hemispheres=[], days=[], align=False, exclude=[]):
  '''
  create a dictionary with sessions_select, all_photo_data_select, pho_scale_together_select, idx_select_sessions,
  z_score_select for different regions/mice/days...
  '''
  dict={}
  if not any(hemispheres):
    hemispheres = [[]] * len(dict_name)
  if not any(days):
    days = [[]] * len(dict_name)
  for i, name in enumerate(dict_name):

    sessions_select, all_photo_data_select, pho_scale_together_select = select_sessions_to_plot(
      sessions_together, all_photo_data_info, pho_scale_together, mice[i], regions[i], hemispheres[i],
      days[i], exclude=exclude)
    idx_select_sessions = select_sessions_to_plot(
      sessions_together, all_photo_data_info, pho_scale_together, mice[i], regions[i], hemispheres[i],
      days[i], return_idx=True, exclude=exclude)

    z_score_select = [stats.zscore(pho_scale_together_select[i], axis=None, ddof=0, nan_policy='omit') for i in
                      range(len(pho_scale_together_select))]

    if align == True:
      all_sample_times_pyc_select = [all_sample_times_pyc[idx] for idx in idx_select_sessions]
      corrected_signal_select = [all_corrected_signal_together[idx] for idx in idx_select_sessions]
      z_score_corrected_signal = [stats.zscore(signal, axis=None, ddof=0) for signal in corrected_signal_select]

      event1 = ['choice_state', 'choice_state', 'choice_state', 'up_state', 'down_state']
      event2 = ['choose_left', 'choose_right', 'reward_consumption', 'choose_up', 'choose_down']
      possible_events = [['choice_state', 'choose_left', 'choose_right'],
                         ['choice_state', 'choose_left', 'choose_right'],
                         ['choice_state', 'reward_consumption'],
                         ['up_state', 'choose_up'],
                         ['down_state', 'choose_down']]

      join_events = {'choice': ['choose_left', 'choose_right'],
                     'second_step_state': ['up_state', 'down_state'],
                     'second_step_choice': ['choose_up', 'choose_down']}

      pre_post_times = {'init_trial': [250, 500],
                        'choice_state': [250, 500],
                        'choice': [500, 1700],
                        'second_step_choice': [1000, 3000]}

      t_aligned = {k:[] for k in pre_post_times.keys()}
      pho_aligned = {k:[] for k in pre_post_times.keys()}
      if i == 0:
        max_len = {k:[] for k in pre_post_times.keys()}

      n = 0
      for session, sample_times_pyc, corrected_signal in zip(sessions_select, all_sample_times_pyc_select, z_score_corrected_signal):
        print('ses:{}, {}'.format(n, name))
        dict_events_id = get_dict_index_session_events(session, event1, event2, possible_events, join_events,
                                                            trial_type='all', selection_type='all', select_n=15,
                                                            block_type='all')
        dict_events_times = get_dict_event_times(session, dict_events_id)

        for state in pre_post_times.keys():
          idx1 = [get_index_time(sample_times_pyc, t1 - pre_post_times[state][0]) for t1 in dict_events_times[state]]
          idx2 = [get_index_time(sample_times_pyc, t1 + pre_post_times[state][1]) for t1 in dict_events_times[state]]

          if max_len[state] == []:
            max_len[state] = max([i2 - i1 for i1, i2 in zip(idx1, idx2)])
          _, temp_pho_scale = zip(
            *[scale_time_events(pre_post_times[state][0] + pre_post_times[state][1], max_len[state], corrected_signal, p1, p2) for p1, p2 in
              zip(idx1, idx2)])
          _, temp_t_scale = zip(
            *[scale_time_events(pre_post_times[state][0] + pre_post_times[state][1], max_len[state], sample_times_pyc, p1, p2) for p1, p2 in
              zip(idx1, idx2)])
          t_aligned[state].append(temp_t_scale)
          pho_aligned[state].append(temp_pho_scale)
        n += 1

      dict[name] = {'mice': mice[i],
                    'region': regions[i],
                    'hemisphere': hemispheres[i],
                    'days': days[i],
                    'sessions_select': sessions_select,
                    'all_photo_data_select': all_photo_data_select,
                    'pho_scale_together_select': pho_scale_together_select,
                    'idx_select_sessions': idx_select_sessions,
                    'z_score_select': z_score_select,
                    'corrected_signal_select': corrected_signal_select,
                    'z_score_corrected_signal': z_score_corrected_signal,
                    'all_sample_times_pyc_select': all_sample_times_pyc_select,
                    't_aligned': t_aligned,
                    'pho_aligned': pho_aligned}
    else:
      dict[name] = {'mice': mice[i],
                    'region': regions[i],
                    'hemisphere': hemispheres[i],
                    'days': days[i],
                    'sessions_select': sessions_select,
                    'all_photo_data_select': all_photo_data_select,
                    'pho_scale_together_select': pho_scale_together_select,
                    'idx_select_sessions': idx_select_sessions,
                    'z_score_select': z_score_select}
  return dict

#%% Plotting
def get_id_trials_to_analyse_all_sessions(sessions, selection_type, select_n, block_type, trial_type, hemisphere='L',
                                          extra_variable=[], extra_variable_name=[]):
  '''
  return list with the index of the trials selected per session
  '''
  hemisphere_list = [hemisphere[i] for i in range(len(sessions))]
  extra_variable_list = [[ep[i] for ep in extra_variable] for i in range(len(sessions))]
  results = \
    pp.starmap(select_trial_types_to_analyse, zip(sessions, [selection_type] * len(sessions), [select_n] * len(sessions),
                                                             [block_type] * len(sessions),
                                                             hemisphere_list, extra_variable_list,
                                                  [extra_variable_name] * len(sessions)))

  trial_id_sessions = [results[i][trial_type] for i in range(len(results))]

  return trial_id_sessions

def trial_type_signal_subjects(subjects, sessions, all_photo_data, pho_scale_together, selection_type, select_n,
                               block_type, trial_type, forced_choice=True, extra_variable=[], extra_variable_name=[]):
  '''
  return mean photometry signal across a trial for the selected trial type per subject
  '''
  all_sub_pho_scale_trial_type_mean = []
  all_sub_pho_scale_trial_type_sem = []
  for sub in subjects:
    idx_sub = np.where([all_photo_data[i]['subject_ID'] in sub for i in range(len(all_photo_data))])[0]
    sessions_sub = [sessions[x] for x in idx_sub]
    pho_scale_together_sub = [pho_scale_together[x] for x in idx_sub]
    hemisphere_sub = [all_photo_data[x]['hemisphere'] for x in idx_sub]
    if extra_variable != []:
      extra_variable_sub = [[ev[x] for x in idx_sub] for ev in extra_variable]
    else:
      extra_variable_sub = []

    trial_id_sessions = get_id_trials_to_analyse_all_sessions(sessions_sub, selection_type, select_n, block_type,
                                                              trial_type, hemisphere_sub, extra_variable=extra_variable_sub,
                                                              extra_variable_name=extra_variable_name)

    free_choice_id_sessions = get_id_trials_to_analyse_all_sessions(sessions_sub, selection_type, select_n, block_type,
                                                              'free_choice', hemisphere_sub)

    if forced_choice == True:
      pho_scale_trial_type = np.asarray([pho_scale_together_sub[s][x] for s in range(len(pho_scale_together_sub))
                                         for x in trial_id_sessions[s]])
    elif forced_choice == 'only':
      pho_scale_trial_type = np.asarray([pho_scale_together_sub[s][x] for s in range(len(pho_scale_together_sub))
                                         for x in trial_id_sessions[s] if x not in free_choice_id_sessions[s]])
    else:
      pho_scale_trial_type = np.asarray([pho_scale_together_sub[s][x] for s in range(len(pho_scale_together_sub))
                                         for x in trial_id_sessions[s] if x in free_choice_id_sessions[s]])

    all_sub_pho_scale_trial_type_mean.append(np.nanmean(pho_scale_trial_type, axis=0))
    all_sub_pho_scale_trial_type_sem.append(stats.sem(pho_scale_trial_type, axis=0, nan_policy='omit'))

  return all_sub_pho_scale_trial_type_mean, all_sub_pho_scale_trial_type_sem

def plot_scaled_trials(sessions, all_photo_data, t_scale_whole, pho_scale_together, v_line, time_start, all_trial_type, **kwargs):
  '''
  Plot time-warped photometry signals across a trial of the selected trial types. Plots mean and SEM across subjects
  '''

  selection_type = kwargs.pop('selection_type', 'end')
  select_n = kwargs.pop('select_n', 15)
  block_type = kwargs.pop('block_type', 'all')
  axvline = kwargs.pop('axvline', [])
  xlabel = kwargs.pop('xlabel', [])
  legend = kwargs.pop('legend', [])
  old_task = kwargs.pop('old_task', False)
  all_event_lines = kwargs.pop('all_event_lines', False)
  plot_legend = kwargs.pop('plot_legend', False)
  colors = kwargs.pop('colors', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'gold', 'salmon', 'hotpink', 'teal', 'darkorange', 'peru', 'aquamarine'])
  line_style = kwargs.pop('line_style', ['-', '-', '-', '-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-'])
  subjects_analyse = kwargs.pop('subjects_analyse', 'all')
  subjects_list = kwargs.pop('subjects_list', None)
  save = kwargs.pop('save', [False, 'folder', 'name'])
  scaled = kwargs.pop('scaled', True)
  forced_choice = kwargs.pop('forced_choice', True)
  extra_variable = kwargs.pop('extra_variable', [])
  extra_variable_name = kwargs.pop('extra_variable_name', [])
  ax = kwargs.pop('ax', None)
  ylim=kwargs.pop('ylim', None)

  ax = ax or plt.gca()
  if subjects_analyse == 'all':
    subjects_lst = [list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))]))]
  elif subjects_analyse == 'all_select':
    subjects_lst = [list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))
                              if all_photo_data[i]['subject_ID'] in map(str, subjects_list)]))]
  else:
    subjects_lst = sorted([[sub] for sub in list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))]))])
  for subplot, subjects in enumerate(subjects_lst):
    for c, trial_type in enumerate(all_trial_type):
      if len(subjects_lst) > 1:
        ax = plt.subplot(int(np.ceil(len(subjects_lst)/3)), 3, subplot + 1)
        plt.title(subjects)
      all_sub_pho_scale_trial_type_mean, all_sub_pho_scale_trial_type_sem = trial_type_signal_subjects(
        subjects, sessions, all_photo_data, pho_scale_together, selection_type, select_n, block_type, trial_type,
        forced_choice, extra_variable=extra_variable, extra_variable_name=extra_variable_name)

      ax.plot(t_scale_whole - time_start, np.nanmean(all_sub_pho_scale_trial_type_mean, axis=0), color=colors[c], ls=line_style[c])
      if len(subjects) > 1:
        sem = stats.sem(all_sub_pho_scale_trial_type_mean, axis=0, nan_policy='omit')
      else:
        sem = all_sub_pho_scale_trial_type_sem[0]
      ax.fill_between(t_scale_whole - time_start, np.nanmean(all_sub_pho_scale_trial_type_mean, axis=0) + sem,
                       np.nanmean(all_sub_pho_scale_trial_type_mean, axis=0) - sem, alpha=0.5, facecolor=colors[c])

    if scaled == True:
      if old_task:
        v_line_param = {0: [[v_line[0] - time_start, 'centre poke', -0.01]],
                        1: [[v_line[1] - time_start, 'choice poke L/R', 0.02],
                            [v_line[2] - time_start, 'second-step state', -0.01]],
                        2: [[v_line[3] - time_start, 'poke U/D', -0.01],
                            [v_line[3] - time_start + 200, 'reward cue (high/low freq) / white noise', -0.02],
                            [v_line[3] - time_start + 500, 'end cue / reward delivery', -0.01]]}
      else:
        v_line_param = {0: [[v_line[0] - time_start, 'centre poke', -0.01]],
                        1: [[v_line[1] - time_start, 'choice poke L/R', 0.02],
                            [v_line[1] - time_start + 200, 'cue second step (high/low freq)', -0.01],
                            [v_line[1] - time_start + 1200, 'end cue', -0.02]],
                        2: [[v_line[3] - time_start, 'poke U/D', -0.01],
                            [v_line[3] - time_start + 200, 'reward cue (high/low freq) / white noise', -0.02],
                            [v_line[3] - time_start + 700, 'end cue / reward delivery', -0.01]]}

      if all_event_lines == True:
        [ax.axvline(v_line_param[x][i][0], linestyle='--', color='k', linewidth=1) for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]

        if plot_legend == True:
          bbox_props = dict(boxstyle="Square", fc="w", ec="0.5", alpha=0.9)
          ax.annotate('\n'.join([str(v_line_param[x][i][0]) + ' ms - ' + v_line_param[x][i][1]
                                  for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]), (0.5, 0),
                       ha='left', va='bottom', xycoords=('axes fraction', 'figure fraction'), annotation_clip=False,
                       bbox=bbox_props, fontsize=10)
        else:
          ax.set_xticks([v_line_param[0][0][0], v_line_param[1][0][0], v_line_param[2][0][0], v_line_param[2][2][0]])
          ax.set_xticklabels(['I', 'C', 'SS', 'O'])
      else:
        [ax.axvline(axv, linestyle='--', color='k', linewidth=1) for axv in [v_line_param[0][0][0], v_line_param[1][0][0], v_line_param[2][0][0],
                                                 v_line_param[2][2][0]]]
        ax.set_xticks([v_line_param[0][0][0], v_line_param[1][0][0], v_line_param[2][0][0], v_line_param[2][2][0]])
        ax.set_xticklabels(['I', 'C', 'SS', 'O'])
    else:
      [ax.axvline(axv[0], linestyle='--', color='k') for axv in v_line]
      ax.set_xticks([vl[0] for vl in v_line])
      ax.set_xticklabels([vn[1] for vn in v_line])

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('z-score')
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
    colors = [colors[i] for i in range(len(all_trial_type))]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    ax.margins(0, 0.05)
  if ylim:
    ax.set_ylim(ylim)
  if legend == []:
    labels = all_trial_type
  else:
    labels = legend
  if plot_legend == True:
    ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.3, -0.1), fancybox=False, shadow=False, ncol=1)
  plt.gcf().set_tight_layout(True)

  if save[0] == True:
    pl.savefig(save[1], save[2])

def _compute_statsmodel_linear_regression_timepoint(all_pho_scale_timepoint, all_predictors):
  '''
  compute linear regression of the photometry data using statsmodels.OLS
  '''
  X = sm.add_constant(all_predictors)
  model = sm.OLS(all_pho_scale_timepoint, X, missing='drop')
  estimates = model.fit()
  coef = estimates.summary2().tables[1]['Coef.'][1:].values
  std = estimates.summary2().tables[1]['Std.Err.'][1:].values
  intercept = estimates.summary2().tables[1]['Coef.'][0]
  std_intercept = estimates.summary2().tables[1]['Std.Err.'][0]
  pval = estimates.summary2().tables[1]['P>|t|'][1:].values
  pval_intercept = estimates.summary2().tables[1]['P>|t|'][0]

  aov_table = anova_lm(model, typ=2)
  eta_squared = aov_table[:-1]['sum_sq']/sum(aov_table['sum_sq'])

  return coef, std, pval, intercept, std_intercept, pval_intercept, eta_squared

def _get_session_photometry_predictors(session, pho_scale_together_i, selection_type, select_n, block_type,
                                       base_predictors, lags, single_lag, forced_choice, fits, fits_names, hemisphere,
                                       Q_values, extra_predictors, timed_data, timed_data_names):
  '''
  return photometry data and predictors of the trials to analyse
  '''
  data_to_analyse = pl._get_data_to_analyse(session, transform_rew_state=False)

  choices, predictors = pl._get_predictors(data_to_analyse, base_predictors, lags, single_lag, session,
                                           fits=fits, fits_names=fits_names, hemisphere=hemisphere, Q_values=Q_values,
                                           extra_predictors=extra_predictors, timed_data_names=timed_data_names)

  # select trials to analyse
  dict_events_id = select_trial_types_to_analyse(session, selection_type=selection_type, select_n=select_n,
                                                 block_type=block_type)

  if forced_choice == True:
    pho_scale = [pho_scale_together_i[x] for x in dict_events_id['all']]
    choices = [choices[x] for x in dict_events_id['all']]
    predictors = [predictors[x] for x in dict_events_id['all']]
    if timed_data != []:
      timed_data_predictor = [[td[x] for x in dict_events_id['all']] for td in timed_data]
    else:
      timed_data_predictor = []
  elif forced_choice == 'only':
    # analyse only forced choice trials
    pho_scale = [pho_scale_together_i[x] for x in dict_events_id['all'] if x not in dict_events_id['free_choice']]
    choices = [choices[x] for x in dict_events_id['all'] if x not in dict_events_id['free_choice']]
    predictors = [predictors[x] for x in dict_events_id['all'] if x not in dict_events_id['free_choice']]
    if timed_data != []:
      timed_data_predictor = [[td[x] for x in dict_events_id['all'] if x not in dict_events_id['free_choice']] for td in timed_data]
    else:
      timed_data_predictor = []
  else:
    #eliminate forced choice trials
    pho_scale = [pho_scale_together_i[x] for x in dict_events_id['all'] if x in dict_events_id['free_choice']]
    choices = [choices[x] for x in dict_events_id['all'] if x in dict_events_id['free_choice']]
    predictors = [predictors[x] for x in dict_events_id['all'] if x in dict_events_id['free_choice']]
    if timed_data != []:
      timed_data_predictor = [[td[x] for x in dict_events_id['all'] if x in dict_events_id['free_choice']] for td in timed_data]
    else:
      timed_data_predictor = []

  return pho_scale, predictors, timed_data_predictor

def _compute_photometry_regression(sessions, pho_scale_together, selection_type,
                                   select_n, block_type, base_predictors, lags, single_lag, forced_choice=False,
                                   sum_predictors=False, fits=[], fits_names=[], hemisphere=[], plot_heatmap=True,
                                   regression_type='Linear', plot_correlation=False, Q_values=[], extra_predictors=[],
                                   timed_data=[], timed_data_names=[], return_predictors_array=False,
                                   zscoring_variables=[]):
  '''
  Compute photometry regression across each trial timepoint
  regression_type = 'Linear', 'Lasso', 'LassoCV', 'OLS'
  '''

  all_predictors = []
  all_timed_predictors = []
  all_pho_scale = []

  if type(sum_predictors) is list:
    sum_predictors = [[x - 1 for x in sum_predictors[i]] for i in range(len(sum_predictors))]
    sum_predictors1 = sum_predictors[:]
    lags_base = [lags * i for i in range(1, len(base_predictors))]
    for lb in lags_base:
      sum_predictors1 += [[x + lb for x in sum_predictors[i]] for i in range(len(sum_predictors))]

  for i, session in enumerate(sessions):
    if Q_values != []:
      Q_val_session = Q_values[i]
    else:
      Q_val_session = []
    if fits != []:
      fits_session = fits[i]
    else:
      fits_session = []
    if extra_predictors != []:
      extra_predictors_session = [[ep[0], ep[1][i]] for ep in extra_predictors]
    else:
      extra_predictors_session = []
    if timed_data != []:
      timed_data_session = [td[i] for td in timed_data]
    else:
      timed_data_session = []

    pho_scale, predictors, timed_data_predictors = _get_session_photometry_predictors(session=session,
                                                               pho_scale_together_i=pho_scale_together[i],
                                                               selection_type=selection_type, select_n=select_n,
                                                               block_type=block_type, base_predictors=base_predictors,
                                                               lags=lags, single_lag=single_lag,
                                                               forced_choice=forced_choice, fits=fits_session,
                                                               fits_names=fits_names, hemisphere=hemisphere[i],
                                                               Q_values=Q_val_session,
                                                               extra_predictors=extra_predictors_session,
                                                               timed_data=timed_data_session, timed_data_names=timed_data_names)

    if type(sum_predictors) is list:
      predictors = [[np.sum([predictors[t][x] for x in sum_predictors1[i]]) for i in range(len(sum_predictors1))]
                    for t in range(len(predictors))]

    if all_predictors != []:
      all_predictors += predictors
      all_pho_scale += pho_scale
      for tni in range(len(timed_data_names)):
        all_timed_predictors[tni] += timed_data_predictors[tni][:]
    else:
      all_predictors = predictors[:]
      all_pho_scale = pho_scale[:]
      all_timed_predictors = timed_data_predictors[:]

  if (zscoring_variables != []) and (zscoring_variables[0] != 'all'):
    all_predictors = np.asarray(all_predictors)
    for name in zscoring_variables:
      all_predictors[:, base_predictors.index(name)] = stats.zscore(all_predictors[:, base_predictors.index(name)])

  if return_predictors_array is True:
    return all_predictors, all_pho_scale

  # Linear regression
  if regression_type != 'OLS':
    if regression_type == 'Linear':
      log_reg = lm.LinearRegression()
    elif regression_type == 'Lasso':
      log_reg = lm.Lasso(alpha=0.001)
    elif regression_type == 'LassoCV':
      log_reg = lm.MultiTaskLassoCV(random_state=0, max_iter=100000)

    if timed_data != []:
      non_nan_val = 0
      interp_val = 0
      all_coef = []
      all_intercept = []
      idx = [base_predictors.index(tdn) for tdn in timed_data_names]
      sort_idx = np.asarray(idx)[np.argsort(idx)]
      timed_predictors = np.asarray(all_timed_predictors)[np.argsort(idx)]
      for tidx in range(len(timed_predictors)):
        for tp in timed_predictors[tidx]:
          if np.isnan(tp).all():
            non_nan_val += 1
            nans, x = nan_helper(tp)
            tp[nans] = 0
          elif np.isnan(tp).any():
            interp_val += 1
            nans, x = nan_helper(tp)
            tp[nans] = np.interp(x(nans), x(~nans), tp[~nans])

      for pi, pho in enumerate(np.asarray(all_pho_scale).T):
        pred = all_predictors[:]
        for si, tpi in zip(sort_idx, timed_predictors[:, :, pi]):
          pred = np.insert(np.asarray(pred), si, tpi, axis=1) # double check this is doing the correct thing
        if np.isnan(pred).any():
          print('NaN in predictors')
        else:
          log_reg.fit(pred, pho)
          if all_coef != []:
            all_coef += [log_reg.coef_]
            all_intercept += [log_reg.intercept_]

          else:
            all_coef = [log_reg.coef_]
            all_intercept = [log_reg.intercept_]

    else:
      idx_nan = np.where(np.isnan(all_pho_scale))[0]
      all_pho_scale = [all_pho_scale[i] for i in range(len(all_pho_scale)) if i not in idx_nan]
      all_predictors = [all_predictors[i] for i in range(len(all_predictors)) if i not in idx_nan]

      if (zscoring_variables != []) and (zscoring_variables[0] == 'all'):
        # standardise regressors
        scaler = StandardScaler()
        all_predictors = scaler.fit_transform(all_predictors)

      log_reg.fit(all_predictors, all_pho_scale)

      predicted_values = log_reg.predict(all_predictors)
      residuals = all_pho_scale - predicted_values

      print(log_reg.score(all_predictors, all_pho_scale))
      if regression_type == 'LassoCV':
        alpha = log_reg.alpha_
        print('alpha: {}'.format(log_reg.alpha_))
      else:
        alpha = []

    if plot_correlation == True:
      if (len(base_predictors) > 1 and (plot_heatmap == True)):
        plt.figure()
        plt.imshow(np.corrcoef(np.asarray(all_predictors).T))
        plt.xticks(range(len(base_predictors)), base_predictors, fontsize=12, rotation=90)
        plt.yticks(range(len(base_predictors)), base_predictors, fontsize=12)
        plt.colorbar()
        plt.gcf().set_tight_layout(True)

    if len(base_predictors)>1:
      corr1 = np.asarray(base_predictors)[np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[0][np.not_equal(
        np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[0], np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[1])]]
      corr2 = np.asarray(base_predictors)[np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[1][np.not_equal(
        np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[0], np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[1])]]

      if corr1.size:
        for icorr in range(len(corr1)):
          warnings.warn('{} and {} are >0.7 correlated'.format(corr1[icorr], corr2[icorr]))

    rank = np.linalg.matrix_rank(all_predictors)
    if regression_type not in ['LassoCV', 'Lasso']:
      if log_reg.rank_ > len(base_predictors):
        warnings.warn('The matrix rank is {}'.format(log_reg.rank_))

    if timed_data != []:
      if non_nan_val > 0:
        print('{} trials did not have non-nan value: predictor with 0'.format(non_nan_val))
      if interp_val > 0:
        print('{} trials contained NaN values: predictor was interpolated'.format(interp_val))
      return all_coef, all_intercept, [], residuals
    else:
      return log_reg.coef_, log_reg.intercept_, alpha, residuals
  else:
    pp.enable_multiprocessing()
    coef, std, pval, intercept, std_intercept, pval_intercept, eta_squared = zip(*pp.map(partial(
      _compute_statsmodel_linear_regression_timepoint, all_predictors=all_predictors), np.asarray(all_pho_scale).T))
    pp.disable_multiprocessing()
    return coef, std, pval, intercept, std_intercept, pval_intercept, eta_squared

def plot_photometry_regression(sessions, all_photo_data, t_scale_whole, pho_scale_together, v_line, time_start,
                               selection_type, select_n, block_type, base_predictors, lags={}, single_lag={},
                               forced_choice=False, sum_predictors=False, title=[], text_box=False,
                               fits=[], all_event_lines=False, plot_legend=True, return_coef=False,
                               regression_type='Linear', subplots=False,ttest=False, multitest=False, test_visual='mask',
                               subsampled_ttest=1, figsize=(5,10),
                               shaded_area=[], plot_correlation=False, Q_values=[], extra_predictors=[], plot_intercept=True, **kwargs):
  '''
  Plot photometry regression
  Q_values: list of lists - [[Q value name, Q_values],[...]]
  '''
  colors = kwargs.pop('colors', ['C0', 'darkred', 'C2', 'C3', 'C4', 'C5', 'C6', 'indigo', 'C8', 'C9', 'b', 'orange', 'gold',
                                 'aqua', 'palegoldenrod', 'black', 'slategray', 'crimson',
                                  'khaki',
            'C0', 'darkred', 'C2', 'C3', 'C4', 'C5', 'C6', 'indigo', 'C8', 'C9', 'b'])
  subjects_list = kwargs.pop('subjects_list', False)
  timed_data = kwargs.pop('timed_data', [])
  per_subject = kwargs.pop('per_subject', True)
  effect_size = kwargs.pop('effect_size', False)
  zscoring_variables = kwargs.pop('zscoring_variables', [])
  plot_residuals = kwargs.pop('plot_residuals', False)

  if subjects_list == False:
    subjects = list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))]))
  else:
    subjects = ['{}'.format(sl) for sl in subjects_list]
    subjects = [sub for sub in subjects if sub in list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))]))]
  coef_reg_all = []
  std_reg_all = []
  pval_reg_all = []
  eta_squared_all = []
  alpha_ls_all = []
  residuals_all = []
  if per_subject == False:
    subjects = [subjects]
  for sub in subjects:
    print(sub)
    idx_sub = np.where([all_photo_data[i]['subject_ID'] in sub for i in range(len(all_photo_data))])[0]
    sessions_sub = [sessions[x] for x in idx_sub]
    hemisphere = [all_photo_data[x]['hemisphere'] for x in idx_sub]
    pho_scale_together_sub = [pho_scale_together[x] for x in idx_sub]
    if Q_values != []:
      Q_values_sub = [Q_values[x] for x in idx_sub]
    else:
      Q_values_sub = []
    if fits != []:
      fits_sub = [fits[1][x] for x in idx_sub]
      fits_names = fits[0]
    else:
      fits_sub = []
      fits_names = []
    if extra_predictors != []:
      extra_predictors_sub = [[ep[0], [ep[1][x] for x in idx_sub]] for ep in extra_predictors]
    else:
      extra_predictors_sub = []
    if timed_data != []:
      timed_data_names = timed_data[0]
      timed_data_sub = [[td[x] for x in idx_sub] for td in timed_data[1]]
    else:
      timed_data_sub = []
      timed_data_names = []

    if regression_type != 'OLS':
      coef_reg,  coef_intercept, alpha_ls, residuals = _compute_photometry_regression(sessions=sessions_sub,
                                                                 pho_scale_together=pho_scale_together_sub,
                                                                 selection_type=selection_type, select_n=select_n,
                                                                 block_type=block_type, base_predictors=base_predictors,
                                                                 lags=lags, single_lag=single_lag,
                                                                 forced_choice=forced_choice,
                                                                 sum_predictors=sum_predictors, fits=fits_sub, fits_names=fits_names,
                                                                 hemisphere=hemisphere, regression_type=regression_type,
                                                                 plot_correlation=plot_correlation, Q_values=Q_values_sub,
                                                                 extra_predictors=extra_predictors_sub, timed_data=timed_data_sub,
                                                                 timed_data_names=timed_data_names, zscoring_variables=zscoring_variables)

      if plot_intercept:
        coef_reg_all.append(np.append(coef_reg, [[x] for x in coef_intercept], axis=1))
      else:
        coef_reg_all.append(coef_reg)
      alpha_ls_all.append(alpha_ls)
      residuals_all.append(residuals)
    else:
      coef, std, pval, intercept, std_intercept, pval_intercept, eta_squared = _compute_photometry_regression(sessions=sessions_sub,
                                                                                                 pho_scale_together=pho_scale_together_sub,
                                                                                                 selection_type=selection_type,
                                                                                                 select_n=select_n,
                                                                                                 block_type=block_type,
                                                                                                 base_predictors=base_predictors,
                                                                                                 lags=lags,
                                                                                                 single_lag=single_lag,
                                                                                                 forced_choice=forced_choice,
                                                                                                 sum_predictors=sum_predictors,
                                                                                                 fits=fits_sub,
                                                                                                 fits_names=fits_names,
                                                                                                 hemisphere=hemisphere,
                                                                                                 regression_type=regression_type,
                                                                                                 plot_correlation=plot_correlation,
                                                                                                 Q_values=Q_values_sub,
                                                                                                 extra_predictors=extra_predictors_sub,
                                                                                                 timed_data=timed_data_sub,
                                                                                                 timed_data_names=timed_data_names)
      if plot_intercept:
        coef_reg_all.append(np.append(coef, [[x] for x in intercept], axis=1))
        std_reg_all.append(np.append(std, [[x] for x in std_intercept], axis=1))
        pval_reg_all.append(np.append(pval, [[x] for x in pval_intercept], axis=1))
      else:
        coef_reg_all.append(coef)
        std_reg_all.append(std)
        pval_reg_all.append(pval)
        eta_squared_all.append(eta_squared)

  if plot_intercept:
    base_predictors = base_predictors + ['intercept']
    if subplots:
      subplots = subplots + [['intercept']]
  if (len(subjects) == 1) or (per_subject == False):
    mean = coef_reg_all[0]
    if regression_type == 'OLS':
      sem = std_reg_all[0] # instead of sem plot std
      std = std_reg_all[0] # instead of sem plot std
      eta_sq = eta_squared_all[0]
    else:
      sem = []
      std=[]
      eta_sq=[]
  else:
    mean = np.mean(coef_reg_all, axis=0)
    sem = stats.sem(coef_reg_all, axis=0)
    if regression_type == 'OLS':
      std = np.mean(std_reg_all, axis=0)
      eta_sq = np.mean(eta_squared_all, axis=0)
      print(eta_sq)
    else:
      std = np.std(coef_reg_all, axis=0, ddof=1) #unbiased estimate of the std
      eta_sq = []

  if ttest == True:

    t, prob = stats.ttest_1samp([coef_reg_all[i][::subsampled_ttest] for i in range(len(coef_reg_all))], 0, axis=0)


  if all_event_lines != 'custom':

    v_line_param = {0: [[v_line[0] - time_start, 'centre poke', -0.01]],
                    1: [[v_line[1] - time_start, 'choice poke L/R', 0.02],
                        [v_line[1] - time_start + 200, 'cue second step (high/low freq)', -0.01],
                        [v_line[1] - time_start + 1200, 'end cue', -0.02]],
                    2: [[v_line[3] - time_start, 'poke U/D', -0.01],
                        [v_line[3] - time_start + 200, 'reward cue (high/low freq) / white noise', -0.02],
                        [v_line[3] - time_start + 700, 'end cue / reward delivery', -0.01]]}

  if subplots == False:

    plt.figure(figsize=figsize)
    for i in range(len(base_predictors)):
      if (ttest == False) or (test_visual == 'dots'):
        plt.plot(t_scale_whole-time_start, mean.T[i], color=colors[i], label=base_predictors[i])
      else:
        plt.plot(t_scale_whole-time_start, mean.T[i], color='k')
        if multitest == True:
          plt.plot((t_scale_whole - time_start)[::subsampled_ttest], np.ma.masked_where(statsmodels.stats.multitest.multipletests(
            prob.T[i], method='fdr_bh')[0] == False, mean.T[i][::subsampled_ttest]), color=colors[i], label=base_predictors[i])
        else:
          plt.plot((t_scale_whole-time_start)[::subsampled_ttest], np.ma.masked_where(prob.T[i] > 0.05, mean.T[i][::subsampled_ttest]), color=colors[i], label=base_predictors[i])

      if (len(subjects) > 1) or (regression_type=='OLS'):
        if (ttest == False) or (test_visual == 'dots'):
          plt.fill_between(t_scale_whole - time_start, mean.T[i] + sem.T[i], mean.T[i] - sem.T[i], alpha=0.5, facecolor=colors[i])
        else:
          plt.fill_between(t_scale_whole - time_start, mean.T[i] + sem.T[i], mean.T[i] - sem.T[i], alpha=0.5, facecolor='k')
          if multitest == True:
            plt.fill_between((t_scale_whole - time_start)[::subsampled_ttest], np.ma.masked_where(
              statsmodels.stats.multitest.multipletests(prob.T[i], method='fdr_bh')[0] == False,
              mean.T[i][::subsampled_ttest]) + sem.T[i][::subsampled_ttest], np.ma.masked_where(
              statsmodels.stats.multitest.multipletests(prob.T[i], method='fdr_bh')[0] == False,
              mean.T[i][::subsampled_ttest]) - sem.T[i][::subsampled_ttest], alpha=0.5, facecolor=colors[i])
          else:
            plt.fill_between(t_scale_whole - time_start, np.ma.masked_where(
              prob.T[i] > 0.05, mean.T[i][::subsampled_ttest]) + sem.T[i][::subsampled_ttest],
                             np.ma.masked_where(prob.T[i] > 0.05, mean.T[i][::subsampled_ttest]) -
                             sem.T[i][::subsampled_ttest], alpha=0.5, facecolor=colors[i])

    if all_event_lines == True:
      [plt.axvline(v_line_param[x][i][0], linestyle='--', color='k') for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]
      if text_box == True:
        bbox_props = dict(boxstyle="Square", fc="w", ec="0.5", alpha=0.9)
        plt.annotate('\n'.join([str(v_line_param[x][i][0]) + ' ms - ' + v_line_param[x][i][1]
                                for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]), (0.5, 0),
                     ha='left', va='bottom', xycoords=('axes fraction', 'figure fraction'), annotation_clip=False,
                     bbox=bbox_props, fontsize=10)
    elif all_event_lines == 'custom':
      [plt.axvline(axv[0], linestyle='--', color='k') for axv in v_line]
      plt.xticks([vl[0] for vl in v_line], [vn[1] for vn in v_line])
    else:
      [plt.axvline(axv, linestyle='--', color='k') for axv in [v_line_param[0][0][0], v_line_param[1][0][0], v_line_param[2][0][0],
                                               v_line_param[2][2][0]]]

      plt.xticks([v_line_param[0][0][0], v_line_param[1][0][0], v_line_param[2][0][0], v_line_param[2][2][0]],
                 ['I', 'C', 'SS', 'O'])

    plt.ylabel('Regression coefficients')
    plt.xlabel('Time (ms)')
    plt.margins(0, 0.05)
    plt.axhline(0, linestyle='--', color='k')
    if title != []:
      plt.title(title)
    if plot_legend == True:
      plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol=2, fontsize=10)
    plt.gcf().set_tight_layout(True)

  else:
    fig, axs = plt.subplots(len(subplots), sharex=True, figsize=figsize)
    for sub in range(len(subplots)):

      for sub_pred in subplots[sub]:
        if (ttest == False) or (test_visual == 'dots'):
          axs[sub].plot(t_scale_whole - time_start, mean.T[base_predictors.index(sub_pred)],
                        color=colors[base_predictors.index(sub_pred)], label=sub_pred)

        else:
          axs[sub].plot(t_scale_whole - time_start, mean.T[base_predictors.index(sub_pred)],
                        color='k')

          if multitest == True:
            axs[sub].plot((t_scale_whole - time_start)[::subsampled_ttest], np.ma.masked_where(statsmodels.stats.multitest.multipletests(
              prob.T[base_predictors.index(sub_pred)], method='fdr_bh')[0] == False, mean.T[base_predictors.index(sub_pred)][::subsampled_ttest]),
                          color=colors[base_predictors.index(sub_pred)], label=sub_pred)
          else:
            axs[sub].plot((t_scale_whole - time_start)[::subsampled_ttest], np.ma.masked_where(prob.T[base_predictors.index(sub_pred)] > 0.05,
                                                                         mean.T[base_predictors.index(sub_pred)][::subsampled_ttest]),
                          color=colors[base_predictors.index(sub_pred)], label=sub_pred)

        if (len(subjects) > 1) or (regression_type=='OLS'):
          if (ttest == False) or (test_visual == 'dots'):
            axs[sub].fill_between(t_scale_whole - time_start, mean.T[base_predictors.index(sub_pred)] +
                             sem.T[base_predictors.index(sub_pred)], mean.T[base_predictors.index(sub_pred)] -
                             sem.T[base_predictors.index(sub_pred)], alpha=0.5,
                             facecolor=colors[base_predictors.index(sub_pred)])
          else:
            axs[sub].fill_between(t_scale_whole - time_start, mean.T[base_predictors.index(sub_pred)] +
                                  sem.T[base_predictors.index(sub_pred)], mean.T[base_predictors.index(sub_pred)] -
                                  sem.T[base_predictors.index(sub_pred)], alpha=0.2,
                                  facecolor='k')
            if multitest == True:
              axs[sub].fill_between((t_scale_whole - time_start)[::subsampled_ttest], np.ma.masked_where(statsmodels.stats.multitest.multipletests(
                prob.T[base_predictors.index(sub_pred)], method='fdr_bh')[0] == False, mean.T[base_predictors.index(sub_pred)][::subsampled_ttest]) +
                                    sem.T[base_predictors.index(sub_pred)][::subsampled_ttest], np.ma.masked_where(statsmodels.stats.multitest.multipletests(
                prob.T[base_predictors.index(sub_pred)], method='fdr_bh')[0] == False, mean.T[base_predictors.index(sub_pred)][::subsampled_ttest]) -
                                    sem.T[base_predictors.index(sub_pred)][::subsampled_ttest], alpha=0.5,
                                    facecolor=colors[base_predictors.index(sub_pred)])
            else:
              axs[sub].fill_between((t_scale_whole - time_start)[::subsampled_ttest], np.ma.masked_where(
                prob.T[base_predictors.index(sub_pred)] > 0.05, mean.T[base_predictors.index(sub_pred)][::subsampled_ttest]) +
                                    sem.T[base_predictors.index(sub_pred)][::subsampled_ttest], np.ma.masked_where(
                prob.T[base_predictors.index(sub_pred)] > 0.05, mean.T[base_predictors.index(sub_pred)][::subsampled_ttest]) -
                                    sem.T[base_predictors.index(sub_pred)][::subsampled_ttest], alpha=0.5,
                                    facecolor=colors[base_predictors.index(sub_pred)])

      if all_event_lines == True:
        [axs[sub].axvline(v_line_param[x][i][0], linestyle='--', color='k') for x in range(len(v_line_param)) for i in
         range(len(v_line_param[x]))]
        if text_box == True:
          bbox_props = dict(boxstyle="Square", fc="w", ec="0.5", alpha=0.9)
          axs[sub].annotate('\n'.join([str(v_line_param[x][i][0]) + ' ms - ' + v_line_param[x][i][1]
                                  for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]), (0.5, 0),
                       ha='left', va='bottom', xycoords=('axes fraction', 'figure fraction'), annotation_clip=False,
                       bbox=bbox_props, fontsize=10)

      elif all_event_lines == 'custom':
        [axs[sub].axvline(axv[0], linestyle='--', color='k') for axv in v_line]

        if shaded_area != []:
          axs[sub].axvspan(shaded_area[0], shaded_area[1], color='yellow', alpha=0.2)

        axs[sub].set_xticks([vl[0] for vl in v_line], minor=False)
        axs[sub].set_xticklabels([vn[1] for vn in v_line], fontdict=None, minor=False)

      else:
        [axs[sub].axvline(axv, linestyle='--', color='k') for axv in [v_line_param[0][0][0], v_line_param[1][0][0], v_line_param[2][0][0],
                                                 v_line_param[2][2][0]]]

        if shaded_area != []:
          axs[sub].axvspan(shaded_area[0], shaded_area[1], color='yellow', alpha=0.2)

        axs[sub].set_xticks([v_line_param[0][0][0], v_line_param[1][0][0], v_line_param[2][0][0], v_line_param[2][2][0]], minor=False)
        axs[sub].set_xticklabels(['I', 'C', 'SS', 'O'], fontdict=None, minor=False)

      axs[sub].margins(0, 0.05)
      axs[sub].axhline(0, linestyle='--', color='k')
      if plot_legend == True:
        legend1 = axs[sub].legend(prop={'size': 6})

    # show significance with stars
    if test_visual == 'dots':
      if multitest == True:
        if effect_size:
          for sub in range(len(subplots)):
            ymin, ymax = axs[sub].get_ylim()
            add_y = 0
            for sub_pred in subplots[sub]:
              p_val = statsmodels.stats.multitest.multipletests(
                prob.T[base_predictors.index(sub_pred)], method='fdr_bh')[1]

              cohens_d = mean.T[base_predictors.index(sub_pred)][::subsampled_ttest]/std.T[base_predictors.index(sub_pred)][::subsampled_ttest]
              if sub_pred == 'reward':
                print(mean.T[base_predictors.index(sub_pred)][::subsampled_ttest])
                print(std.T[base_predictors.index(sub_pred)][::subsampled_ttest])
                print(cohens_d)

              marker_size = [40 if ((d > 0.8) and (p<0.05)) else 30 if ((d > 0.5) and (p<0.05)) else 20 if ((d >= 0.2) and (p<0.05)) else 5 if ((d < 0.2) and (p<0.05)) else 0 for d, p in zip(cohens_d, p_val)]

              scatter = axs[sub].scatter((t_scale_whole - time_start)[::subsampled_ttest],
                                         [ymax + add_y] * len(marker_size), s=marker_size,
                                         color=colors[base_predictors.index(sub_pred)], marker='o')
              add_y += (max([max(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]]) - \
                        min([min(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]])) / 7

          p1 = plt.scatter([], [], s=5, marker='o', color='#555555', alpha=0.6)
          p2 = plt.scatter([], [], s=20, marker='o', color='#555555', alpha=0.6)
          p3 = plt.scatter([], [], s=30, marker='o', color='#555555', alpha=0.6)
          p4 = plt.scatter([], [], s=40, marker='o', color='#555555', alpha=0.6)
          if plot_legend == True:
            legend2 = axs[sub].legend((p1, p2, p3, p4), ('d < 0.2', 'd > 0.2', 'd > 0.5', 'd > 0.8'))
            plt.gca().add_artist(legend1)
        else:
          for sub in range(len(subplots)):
            ymin, ymax = axs[sub].get_ylim()
            add_y = 0
            for sub_pred in subplots[sub]:
              p_val = statsmodels.stats.multitest.multipletests(
                prob.T[base_predictors.index(sub_pred)], method='fdr_bh')[1]
              marker_size = [30 if p < 0.001 else 20 if p < 0.01 else 10 if p < 0.05 else 0 for p in p_val]
              scatter = axs[sub].scatter((t_scale_whole - time_start)[::subsampled_ttest], [ymax + add_y] * len(marker_size), s=marker_size,
                               color=colors[base_predictors.index(sub_pred)], marker='o')
              add_y += (max([max(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]]) - \
                        min([min(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]])) / 7

          p1 = plt.scatter([], [], s=10, marker='o', color='#555555', alpha=0.6)
          p2 = plt.scatter([], [], s=20, marker='o', color='#555555', alpha=0.6)
          p3 = plt.scatter([], [], s=30, marker='o', color='#555555', alpha=0.6)
          if plot_legend == True:
            legend2 = axs[sub].legend((p1, p2, p3), ('p<0.05', 'p<0.01', 'p<0.001'))
            plt.gca().add_artist(legend1)

      elif ttest == True:
        for sub in range(len(subplots)):
          ymin, ymax = axs[sub].get_ylim()
          add_y = 0
          handles_all = []
          labels_all = []
          for sub_pred in subplots[sub]:
            marker_size = [70 if p < 0.0001 else 50 if p < 0.001 else 30 if p < 0.01 else 10 if p < 0.05 else 0 for p
                           in prob.T[base_predictors.index(sub_pred)]]
            scatter = axs[sub].scatter((t_scale_whole - time_start)[::subsampled_ttest], [ymax + add_y] * len(marker_size), s=marker_size,
                             color=colors[base_predictors.index(sub_pred)], marker='o')
            add_y += (max([max(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]]) - \
                      min([min(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]])) / 7

        p1 = plt.scatter([], [], s=20, marker='o', color='#555555', alpha=0.6)
        p2 = plt.scatter([], [], s=60, marker='o', color='#555555', alpha=0.6)
        p3 = plt.scatter([], [], s=100, marker='o', color='#555555', alpha=0.6)
        p4 = plt.scatter([], [], s=140, marker='o', color='#555555', alpha=0.6)
        if plot_legend == True:
          plt.gca().add_artist(legend1)

    for ax in axs:
      ax.label_outer()

    if title != []:
      fig.suptitle(title)
    fig.text(0.01, 0.5, 'Regression coefficients', va='center', rotation='vertical')
    fig.subplots_adjust(wspace=0, hspace=0, top=0.95, right=0.99, bottom=0.05, left=0.07)

    if plot_residuals:
      stack_residuals = np.vstack((residuals_all))
      plt.figure(figsize=(6, 6))
      plt.suptitle(title)
      plt.subplot(2, 1, 1)
      plt.hist(np.mean(stack_residuals, axis=1), bins=30, edgecolor='k')
      plt.title('Histogram of Residuals')
      plt.xlabel('Residual')
      plt.ylabel('Frequency')
      plt.subplot(2, 1, 2)
      stats.probplot(np.mean(stack_residuals, axis=1), dist="norm", plot=plt)
      plt.title('Q-Q Plot')

    if return_coef == True:
      return [coef_reg_all[i] for i in range(len(coef_reg_all))], \
             [mean.T[i] for i in range(len(base_predictors))], \
             [std.T[i] for i in range(len(base_predictors))], \
             [sem.T[i] for i in range(len(base_predictors))], t, prob, \
             [alpha_ls_all[i] for i in range(len(coef_reg_all))], \
             [residuals_all[i] for i in range(len(coef_reg_all))]

