# -------------------------------------------------------------------------------------
# Code to simulate behaviour from agents fits
# Marta Blanco-Pozo, 2023
# Adapted from Akam, Costa & Dayan (2015), Simple Plans or Sophisticated Habits? State, Transition and Learning Interactions in the Two-Step Task
# -------------------------------------------------------------------------------------

import os
import numpy as np
from random import randint, random, shuffle
from copy import deepcopy
from functools import partial
import parallel_processing as pp
import itertools


#------------------------------------------------------------------------------------
# Two-step task with fixed transition probabilities
#------------------------------------------------------------------------------------
class Two_step_fixed:
  def __init__(self, include_forced_choice=True, transition_type=0):
    if include_forced_choice is True:
      self.allowed_choice_type = [1, 0, 'free', 'free', 'free', 'free', 'free', 'free']
    else:
      self.allowed_choice_type = ['free', 'free', 'free', 'free', 'free', 'free']
    self.choice_type = sample_without_replacement(self.allowed_choice_type)
    self.com_trans = 0.8
    self.rare_trans = 0.2
    self.rew_probs = np.array([[0.8, 0.2], # down good
                               [0.2, 0.8], # up good
                               [0.5, 0.5]]) # neutral
    self.threshold = 0.75 # Threshold to cross to change block
    self.tau = 8. # Time constant moving average
    self.neutral_block_length = [20,30] # [min, max]
    self.trials_post_threshold = [5,15] # [min, max]
    self.mov_ave_correct = _exp_mov_ave(tau=self.tau, init_value=0.5) # Moving average of agent choices
    self.transition_block = transition_type
    self.reset()

  def reset(self, n_trials=1000):
    self.reward_block = randint(0,2) # 1: up good, 0: down good, 2: neutral
    if self.reward_block == 2:
      self.block_length = randint(*self.neutral_block_length)
    else:
      self.block_length = randint(*self.trials_post_threshold)
    self.block_trials = 0 # Number of trials into current block
    self.curr_trial = 0 # Current trial number
    self.block_transition = False # True if transition criterion reached in current block and passed x trials post-threshold
    self.trials_post_criterion = 0
    self.trial_number = 1 # Current trial number
    self.n_trials = n_trials # Session length
    self.mov_ave_correct.reset()
    self.end_session = False
    self.blocks = {'start_trials' : [0],
                   'end_trials'   : [],
                   'reward_states': [self.reward_block],
                   'transition_states': [self.transition_block],
                   'trial_trans_state': [],
                   'trial_rew_state'  : []}

  def trial(self, choice):
    self.current_choice_type = self.choice_type.next()
    if self.current_choice_type == 'free':
      free_choice = True
      self.mov_ave_correct.update(0.5 if self.reward_block == 2 else int(choice == int(self.reward_block == self.transition_block)))
    else:
      free_choice = False
      choice = self.current_choice_type
    # Update moving average
    second_step = int((choice == _with_prob(self.com_trans)) == self.transition_block)
    self.block_trials += 1
    self.curr_trial += 1
    outcome = int(_with_prob(self.rew_probs[self.reward_block, second_step]))
    # Check if change block type
    if self.reward_block == 2:
      if self.block_trials >= self.block_length:
        self.block_transition = True
    else:
      if self.mov_ave_correct.ave > self.threshold:
        if self.trials_post_criterion >= self.block_length:
          self.block_transition = True
        else:
          self.trials_post_criterion += 1
    self.blocks['trial_trans_state'].append(self.transition_block)
    self.blocks['trial_rew_state'].append(self.reward_block)

    # Block transition
    if self.block_transition:
      self.block_trials = 0
      self.trials_post_criterion = 0
      self.block_transition = False
      old_rew_block = self.reward_block
      while old_rew_block == self.reward_block:
        self.reward_block = randint(0,2)
      if self.reward_block == 2:
        self.block_length = randint(*self.neutral_block_length)
        self.mov_ave_correct.ave = 0.5
      else:
        self.block_length = randint(*self.trials_post_threshold)
        self.mov_ave_correct.ave = 1 - self.mov_ave_correct.ave

      self.blocks['start_trials'].append(self.curr_trial)
      self.blocks['end_trials'].append(self.curr_trial)
      self.blocks['reward_states'].append(self.reward_block)
      self.blocks['transition_states'].append(self.transition_block)

    # End session
    if self.curr_trial >= self.n_trials:
      self.end_session = True
      self.blocks['end_trials'].append(self.curr_trial) # removed +1

    return choice, second_step, outcome, free_choice

class _exp_mov_ave:
  'Exponential moving average class.'
  def __init__(self, tau=None, init_value=0., alpha=None):
    if alpha is None: alpha = 1 - np.exp(-1 / tau)
    self._alpha = alpha
    self._m = 1 - alpha
    self.init_value = init_value
    self.reset()

  def reset(self, init_value=None):
    if init_value:
      self.init_value = init_value
    self.ave = self.init_value

  def update(self, sample):
    self.ave = (self.ave * self._m) + (self._alpha * sample)


def _with_prob(prob):
  'return true / flase with specified probability .'
  return random() < prob

class sample_without_replacement:
  # Repeatedly sample elements from items list without replacement.
  def __init__(self, items):
    self._all_items = items
    shuffle(self._all_items)
    self._next_items = [] + self._all_items

  def next(self):
    if len(self._next_items) == 0:
      shuffle(self._all_items)
      self._next_items += self._all_items
    return self._next_items.pop()

#------------------------------------------------------------------------------------
# Simulation
#------------------------------------------------------------------------------------

class simulated_session():
  def __init__(self, agent, subject_ID, params, n_trials=1000, task=Two_step_fixed()):
    '''Simulate session with current agent and task parameters'''
    self.param_names = agent.param_names
    self.true_params = params
    self.subject_ID = subject_ID
    self.n_trials = n_trials
    choices, second_steps, outcomes, free_choice = agent.session_simulate(task, params, n_trials)
    self.trial_data = {'choices'     : choices,
                       'transitions' : (choices == second_steps).astype(int),
                       'second_steps': second_steps,
                       'outcomes'    : outcomes,
                       'free_choice' : free_choice.astype(bool)}
    self.blocks = deepcopy(task.blocks)
    self.blocks['reward_states'] = np.asarray(self.blocks['reward_states'])
    self.blocks['transition_states'] = np.asarray(self.blocks['transition_states']).astype(bool)
    self.blocks['trial_trans_state'] = np.asarray(self.blocks['trial_trans_state'])
    self.blocks['trial_rew_state'] = np.asarray(self.blocks['trial_rew_state'])
    self.reward_loc = 'UD'
    self.file_name = 'NaN'
    self.experiment_name = 'simulated'


  def select_trials(self, selection_type, select_n=20, block_type='all'):
    ''' Select specific trials for analysis. - two_step task

    The first selection step is specified by selection_type:

    'end' : Only final select_n trials of each block are selected.

    'xtr' : Select all trials except select_n trials following transition reversal.

    'all' : All trials are included.

    The first_n_mins argument can be used to select only trials occuring within
    a specified number of minutes of the session start.

    The block_type argument allows additional selection for only 'neutral' or 'non_neutral' blocks.
    '''

    assert selection_type in ['start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'], 'Invalid trial select type.'

    if selection_type == 'xtr':  # Select all trials except select_n following transition reversal only
      trials_to_use = np.ones(self.n_trials, dtype=bool)
      trans_change = np.hstack((
        False, ~np.equal(self.blocks['transition_states'][:-1],
                         self.blocks['transition_states'][1:])))

      start_trials = (self.blocks['start_trials'] +
                      [self.blocks['end_trials'][-1] + select_n])
      for i in range(len(trans_change)):
        if trans_change[i]:
          trials_to_use[start_trials[i]:start_trials[i] + select_n] = False

    if selection_type == 'xtrrw':  # Select all trials except select_n following transition reversal only
      trials_to_use = np.ones(self.n_trials, dtype=bool)
      trans_change = np.hstack((
        False, ~np.equal(self.blocks['transition_states'][:-1],
                         self.blocks['transition_states'][1:])))
      rew_change = np.hstack((
        False, ~np.equal(self.blocks['reward_states'][:-1],
                         self.blocks['reward_states'][1:])))
      start_trials = (self.blocks['start_trials'] +
                      [self.blocks['end_trials'][-1] + select_n])
      for i in range(len(trans_change)):
        if trans_change[i]:
          trials_to_use[start_trials[i]:start_trials[i] + select_n] = False
      for i in range(len(rew_change)):
        if rew_change[i]:
          trials_to_use[start_trials[i]:start_trials[i] + select_n] = False

    elif selection_type == 'xmid':  # select trials in the middle
      trials_to_use = np.ones(self.n_trials, dtype=bool)
      trans_change = np.hstack((
        False, ~np.equal(self.blocks['transition_states'][:-1],
                         self.blocks['transition_states'][1:])))
      rew_change = np.hstack((
        False, ~np.equal(self.blocks['reward_states'][:-1],
                         self.blocks['reward_states'][1:])))
      start_trials = (self.blocks['start_trials'] +
                      [self.blocks['end_trials'][-1] + select_n])
      for i in range(len(trans_change)):
        if trans_change[i]:
          trials_to_use[start_trials[i]:start_trials[i] + select_n] = False
          trials_to_use[start_trials[i] + select_n + select_n:] = False
      for i in range(len(rew_change)):
        if rew_change[i]:
          trials_to_use[start_trials[i]:start_trials[i] + select_n] = False
          trials_to_use[start_trials[i] + select_n + select_n:] = False

    elif selection_type == 'end':  # Select only select_n trials before block transitions.
      trials_to_use = np.zeros(self.n_trials, dtype=bool)
      for b in self.blocks['start_trials'][1:]:
        trials_to_use[b - select_n:b] = True  # ELIMINATED -1 FROM ORIGINAL CODE

    elif selection_type == 'start':  # Select only select_n trials after block transitions.
      trials_to_use = np.zeros(self.n_trials, dtype=bool)
      for b in self.blocks['start_trials'][1:]:
        trials_to_use[b:b + select_n] = True

    elif selection_type == 'start_1':  # Select only select_n trials after block transitions but eliminating the first trial.
      trials_to_use = np.zeros(self.n_trials, dtype=bool)
      for b in self.blocks['start_trials'][1:]:
        trials_to_use[b + 1:b + 1 + select_n] = True

    elif selection_type == 'all':  # Use all trials.
      trials_to_use = np.ones(self.n_trials, dtype=bool)

    if not block_type == 'all':  # Restrict analysed trials to blocks of certain types.
      if block_type == 'neutral':  # Include trials only from neutral blocks.
        block_selection = self.blocks['trial_rew_state'] == 2
      elif block_type == 'non_neutral':  # Include trials only from non-neutral blocks.
        block_selection = self.blocks['trial_rew_state'] != 2
      elif block_type == 'non_neutral_after_neutral':
        temp = self.blocks['trial_rew_state'].copy()
        block_change_id = np.hstack((np.where(temp[1:] != temp[:-1])[0], (len(temp) + 1)))
        for ib in range(len(block_change_id) - 1):
          if temp[block_change_id[ib]] == 2:
            temp[block_change_id[ib] + 1:block_change_id[ib + 1] + 1] = 3
        block_selection = temp == 3
      elif block_type == 'non_neutral_after_non_neutral':
        temp = self.blocks['trial_rew_state'].copy()
        block_change_id = np.hstack((np.where(temp[1:] != temp[:-1])[0], (len(temp) + 1)))
        for ib in range(len(block_change_id) - 1):
          if (temp[block_change_id[ib]] != 2) and (temp[block_change_id[ib] + 1] != 2):
            temp[block_change_id[ib] + 1:block_change_id[ib + 1] + 1] = 3
        block_selection = temp == 3

      trials_to_use = trials_to_use & block_selection

    return trials_to_use

  def unpack_trial_data(self, order='CTSO', dtype=int):
    'Return elements of trial_data dictionary in specified order and data type.'
    o_dict = {'C': 'choices', 'T': 'transitions', 'S': 'second_steps', 'O': 'outcomes'}
    if dtype == int:
      return [self.trial_data[o_dict[i]] for i in order]
    else:
      return [self.trial_data[o_dict[i]].astype(dtype) for i in order]


def sim_sessions_from_pop_fit(agent, population_fit, n_ses=[10], n_trials=[1000],
                              task=Two_step_fixed, include_forced_choice=True, transition_type=1):
  '''Simulate sessions using parameter values drawn from the population distribution specified
  by population_fit. alternatively a dictionary of means and variances for each paramter can be
  specified.
  n_trials: list of number of trials to simulate per subject
  n_ses: list of number of sessions per subject
  transition_type: int or list of transition type per animal
  '''
  assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
  agent_iter = [agent] * np.sum(n_ses)
  subject_ID_iter = list(np.repeat(population_fit['sID'], n_ses))
  n_trials_iter = list(np.repeat(n_trials, n_ses))
  if type(transition_type) == int:
    transition_type = list(np.repeat(transition_type, n_ses))
  else:
    transition_type = list(np.repeat(transition_type, n_ses))
  task_iter = [task(include_forced_choice=include_forced_choice, transition_type=tt) for tt in transition_type]
  population_fit_iter = [population_fit] * np.sum(n_ses)
  sessions = pp.starmap(_sim_func, zip(population_fit_iter, agent_iter, subject_ID_iter, n_trials_iter, task_iter))

  return sessions

def sim_sessions_set_params(agent, param_names, params, n_ses=100, n_trials=350):
  '''
  simulate sessions with defined parameters in params
  '''
  assert param_names == agent.param_names, 'Agent parameters do not match fit.'
  agent_iter = [agent] * n_ses
  n_trials_iter = list(np.repeat(n_trials, n_ses))
  sub_id = ['simulated'] * n_ses
  task_iter = [Two_step_fixed()] * n_ses
  params_iter = [params] * n_ses

  sessions = pp.starmap(_sim_func_all_fits, zip(params_iter, agent_iter, sub_id, n_trials_iter, task_iter))
  return sessions

def _sim_func(population_fit, agent, subject_ID, n_trials, task):
  params = np.random.normal(population_fit['means'][population_fit['sID'].index(subject_ID)],
                            population_fit['var'][population_fit['sID'].index(subject_ID)])
  return simulated_session(agent, subject_ID, params, n_trials, task)

def _sim_func_all_fits(params, agent, subject_ID, n_trials, task):
  return simulated_session(agent, subject_ID, params, n_trials, task)

