import numpy as np
import pandas as pd
from functools import partial
import math

from Code_final_manuscript.code import parallel_processing as pp

def _lag(x, i):  # Apply lag of i trials to array x.
  x_lag = np.zeros(x.shape, x.dtype)
  if i > 0:
      x_lag[i:] = x[:-i]
  else:
      x_lag[:i] = x[-i:]
  return x_lag

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

# 1 True --> left --> orange
# 0 True --> right --> orange
#
# 1 False --> right --> blue
# 0 False --> left --> blue
#
# 2 True --> orange
# 2 False --> blue


def _get_dataframe_session(session, base_predictors, stay_pr=False, opto=True, regressor_type='str', lags=[], subjects_virus=[]):
  if not subjects_virus:
    subjects_virus = {63: 'YFP',
                      64: 'ChR2',
                      65: 'ChR2',
                      66: 'YFP',
                      67: 'ChR2',
                      69: 'YFP',
                      70: 'ChR2',
                      71: 'YFP',
                      72: 'ChR2',
                      73: 'ChR2',
                      74: 'ChR2',
                      75: 'YFP'}

  choices = session.trial_data['choices'].astype(bool)
  choices_int = session.trial_data['choices']
  transitions_AB = session.trial_data['transitions'].astype(bool)
  second_steps = session.trial_data['second_steps'].astype(bool)
  outcomes = session.trial_data['outcomes'].astype(bool)
  start_block = session.select_trials(selection_type='start', select_n=10, block_type='all')
  end_block = session.select_trials(selection_type='end', select_n=10, block_type='all')

  stay = choices[1:] == choices[:-1]

  rew_state = session.blocks['trial_rew_state']
  trans_state = session.blocks['trial_trans_state']  # Trial by trial state of the tranistion matrix (A vs B)
  transitions_CR = transitions_AB == trans_state  # Trial by trial common or rare transitions (True: common; False: rare)
  transition_CR_x_outcome = transitions_CR == outcomes  # True: common and rewarded / rare and unrewarded
                                                        # False: common and unrewarded / rare and rewarded
  forced_choice_trials = ~session.trial_data['free_choice']
  forced_choice_next_trial = forced_choice_trials[1:]
  subject = np.repeat(session.subject_ID, len(choices))
  subject_str = np.asarray(['m{}'.format(sub) for sub in subject])
  try:
    date = np.repeat(session.datetime_string.split()[0], len(choices))
  except AttributeError:
    date = np.repeat(['no_date'], len(choices))

  if opto:
    group = np.repeat(subjects_virus[session.subject_ID], len(choices))
    stim = session.trial_data['stim']
    stim_type = session.trial_data['stim_type']
  else:
    group = np.repeat(np.nan, len(choices))
    stim = []
    stim_type = []

  if type(stim) is not np.ndarray:
    stim = np.repeat(np.nan, len(choices))
    stim_type = np.repeat(np.nan, len(choices))
    same_prob_stim = np.repeat(np.nan, len(choices))
    same_prob_stim_1 = np.repeat(np.nan, len(choices))
  else:
    same_prob_stim = np.asarray([True if l1 == 0 and l2 == 0 else False for l1, l2 in zip(_lag(stim, 1), _lag(stim, 2))])
    same_prob_stim_1 = _lag(same_prob_stim, 1)

  d2 = {}
  if regressor_type == 'str':
    if 'pr_choice' in base_predictors:
      d2['pr_choice'] = np.asarray(['left' if c == True else 'cright' for c in choices[:-1]])
    if 'pr_correct' in base_predictors:
      correct_option = np.asarray([1 if tr == 0 else 0 if tr == 1 else 2 for tr in trans_state ^ rew_state])
      d2['pr_correct'] = np.asarray(['b_correct' if c == co else 'a_neutral' if co == 2 else 'c_incorrect' for c, co in zip(choices_int, correct_option)])[:-1]
    if 'pr_outcome' in base_predictors:
      d2['pr_outcome'] = np.asarray(['rewarded' if o == True else 'unrewarded' for o in outcomes[:-1]])
    if 'pr_trans_CR' in base_predictors:
      d2['pr_trans_CR'] = np.asarray(['common' if t == 1 else 'rare' for t in transitions_CR.astype(int)[:-1]])
    if 'pr_trans_CR_x_out' in base_predictors:
      d2['pr_trans_CR_x_out'] = np.asarray(['1' if t == 1 else '-1' for t in transition_CR_x_outcome.astype(int)[:-1]])
    if 'pr_stim' in base_predictors:
      d2['pr_stim'] = np.asarray(['a_stim' if s == 1 else 'non_stim' if s == 0 else np.nan for s in stim[:-1]])
    if 'pr_stim_type' in base_predictors:
      stim_type_name = stim_type[0]
      d2['pr_stim_type'] = np.asarray([stim_type_name if s == 1 else 'non_stim' for s in stim[:-1]])
    if 'pr_stim_type_nontype' in base_predictors:
      stim_type_name = stim_type[0]
      d2['pr_stim_type_nontype'] = np.asarray([stim_type_name if s == 1 else 'non_stim_{}'.format(stim_type_name) for s in stim[:-1]])
    if 'pr_ss' in base_predictors:
      d2['pr_ss'] = np.asarray(['ssup' if ss == True else 'ssdown' for ss in second_steps[:-1]])
    if 'pr_rew_state' in base_predictors:
      d2['pr_rew_state'] = np.asarray(['upblock' if rs == 1 else 'neutral' if rs == 2 else 'downblock' for rs in rew_state[:-1]]) #1 up block, 0 neutral, -1 down block
    if 'pr_start_end_block' in base_predictors:
      d2['pr_start_end_block'] = np.asarray([1 if sb == True else -1 if eb == True else 0 for sb,eb in zip(start_block[:-1], end_block[:-1])])
    if 'pr_stay' in base_predictors:
      d2['pr_stay'] = np.asarray([1 if s == True else -1 for s in stay])

    if 'pr_common_rew' in base_predictors:
      d2['pr_common_rew'] = np.asarray(
        ['1' if (t == 1) and (o == 1) else '-1' for t, o in zip(transitions_CR.astype(int)[:-1], outcomes[:-1])])
    if 'pr_rare_rew' in base_predictors:
      d2['pr_rare_rew'] = np.asarray(
        ['1' if (t == 0) and (o == 1) else '-1' for t, o in zip(transitions_CR.astype(int)[:-1], outcomes[:-1])])
    if 'pr_common_unrew' in base_predictors:
      d2['pr_common_unrew'] = np.asarray(
        ['1' if (t == 1) and (o == 0) else '-1' for t, o in zip(transitions_CR.astype(int)[:-1], outcomes[:-1])])
    if 'pr_rare_unrew' in base_predictors:
      d2['pr_rare_unrew'] = np.asarray(
        ['1' if (t == 0) and (o == 0) else '-1' for t, o in zip(transitions_CR.astype(int)[:-1], outcomes[:-1])])
  else:
    if 'pr_choice' in base_predictors:
      d2['pr_choice'] = np.asarray([1 if c == True else -1 for c in choices[:-1]])
    if 'pr_correct' in base_predictors:
      correct_option = np.asarray([1 if tr == 0 else 0 if tr == 1 else 2 for tr in trans_state ^ rew_state])
      d2['pr_correct'] = np.asarray([1 if c == co else 0 if co == 2 else -1 for c, co in
                             zip(choices_int, correct_option)])[:-1]
    if 'pr_outcome' in base_predictors:
      d2['pr_outcome'] = np.asarray([1 if o == True else -1 for o in outcomes[:-1]])
    if 'pr_trans_CR' in base_predictors:
      d2['pr_trans_CR'] = np.asarray([1 if t == 1 else -1 for t in transitions_CR.astype(int)[:-1]])
    if 'pr_trans_CR_x_out' in base_predictors:
      d2['pr_trans_CR_x_out'] = np.asarray([1 if t == 1 else -1 for t in transition_CR_x_outcome.astype(int)[:-1]])
    if 'pr_stim' in base_predictors:
      d2['pr_stim'] = np.asarray([1 if s == 1 else -1 for s in stim[:-1]])
    if 'pr_ss' in base_predictors:
      d2['pr_ss'] = np.asarray([1 if ss == True else -1 for ss in second_steps[:-1]])
    if 'pr_rew_state' in base_predictors:
      d2['pr_rew_state'] = np.asarray([1 if rs == 1 else 0 if rs == 2 else -1 for rs in
                               rew_state[:-1]])  # 1 up block, 0 neutral, -1 down block
    if 'pr_start_end_block' in base_predictors:
      d2['pr_start_end_block'] = np.asarray(
      [1 if sb == True else -1 if eb == True else 0 for sb, eb in zip(start_block[:-1], end_block[:-1])])
    if 'pr_stay' in base_predictors:
      d2['pr_stay'] = np.asarray([1 if s == True else -1 for s in stay])

    if 'pr_common_rew' in base_predictors:
      d2['pr_common_rew'] = np.asarray((outcomes.astype(bool) & transitions_CR.astype(bool)) * 0.5)[:-1]
    if 'pr_rare_rew' in base_predictors:
      d2['pr_rare_rew'] = np.asarray((outcomes.astype(bool) & ~transitions_CR.astype(bool)) * 0.5)[:-1]
    if 'pr_common_unrew' in base_predictors:
      d2['pr_common_unrew'] = np.asarray((~outcomes.astype(bool) & transitions_CR.astype(bool)) * 0.5)[:-1]
    if 'pr_rare_unrew' in base_predictors:
      d2['pr_rare_unrew'] = np.asarray((~outcomes.astype(bool) & ~transitions_CR.astype(bool)) * 0.5)[:-1]

  if stay_pr is False:
    d = {'subject': subject,
         'subject_str': subject_str,
         'group': group,
         'date': date,
         'choices': choices,
         'choices_int': choices_int,
         'transitions_AB': transitions_AB,
         'second_steps': second_steps,
         'outcomes': outcomes,
         'trans_state': trans_state,
         'transitions_CR': transitions_CR,
         'transition_CR_x_outcome': transition_CR_x_outcome,
         'forced_choice_trials': forced_choice_trials,
         'rew_state': rew_state,
         'stim_trial': stim,
         'stim_type': stim_type,
         'same_prob_stim': same_prob_stim,
         'same_prob_stim_1': same_prob_stim_1
         }
    return pd.DataFrame(data=d)
  else:
    d3 = {'subject': subject[:-1],
          'subject_str': subject_str[:-1],
          'group': group[:-1],
          'date': date[:-1],
          'stay': stay,
          'forced_choice_next_trial': forced_choice_next_trial,
          # because the stay looks at repeating choice on the next trial, we need to eliminate the trials where there is a forced choice on the next trial
          'same_prob_stim': same_prob_stim[:-1],
          'same_prob_stim_1': same_prob_stim_1[:-1],
          'stim_type': stim_type[:-1]
          }
    d = {**d3, **d2}

    df = pd.DataFrame(data=d)

    if lags:
      lags_shift = np.arange(np.abs(lags))
      for bp in base_predictors:
        predictors = np.zeros(len(df['stay']))
        for lg in lags_shift:
          if lg == 0:
            df['{}_{}'.format(bp, lg)] = df[bp]
          else:
            predictors[lg:] = df[bp][:-lg]
            df['{}_{}'.format(bp, lg)] = predictors

    return df


def _get_dataframe_session_regressors(df_session_data, base_predictors, lags={}, single_lag={}):
  '''
  lags: (int) return regressors from -1 to -lags
  single_lag: (int) return regressors with an specific lag (normally this is set to 1)
  '''


  if type(lags) == int:
    lags = {p: lags for p in base_predictors}

  predictors_lags = []  # predictor names including lags.
  for predictor in base_predictors:
    if predictor in list(lags.keys()):
      for i in range(lags[predictor]):
        predictors_lags.append(predictor + '-' + str(i + 1))  # Lag is indicated by value after '-' in name.
    else:
      predictors_lags.append(predictor)  # If no lag specified, defaults to 0.


  bp_values = {}

  for p in base_predictors:
    choices = df_session_data['choices'].values
    transitions_AB = df_session_data['transitions_AB'].values
    second_steps = df_session_data['second_steps'].values
    outcomes = df_session_data['outcomes'].values
    trans_state = df_session_data['trans_state'].values
    transitions_CR = df_session_data['transitions_CR'].values
    transition_CR_x_outcome = df_session_data['transition_CR_x_outcome'].values
    forced_choice_trials = df_session_data['forced_choice_trials'].values
    rew_state = df_session_data['rew_state'].values
    stim = df_session_data['stim_trial'].values
    correct = [0.5 if tr == 0 else -0.5 if tr == 1 else 0 for tr in trans_state ^ rew_state]

    if p == 'correct':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
      bp_values[p] = correct
    elif p == 'prev_rew_state':  # 0.5, 0, -0.5 for up block, neutral, and down block
      bp_values[p] = rew_state
    elif p == 'correct_stim':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
      bp_values[p] = [x if stim[i] == 1 else 0 for i, x in enumerate(correct)]
    elif p == 'correct_nonstim':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
      bp_values[p] = [x if stim[i] == 0 else 0 for i, x in enumerate(correct)]
    elif p == 'side':  # 0.5, -0.5 for up, down poke reached at second step.
      bp_values[p] = second_steps - 0.5
    elif p == 'choice':  # 0.5, - 0.5 for choices left, right.
      bp_values[p] = choices - 0.5
    elif p == 'prev_choice':  # 0.5, - 0.5 for choices left, right. (same as the predictor on the top but changed name for consistency)
      bp_values[p] = choices - 0.5
    elif p == 'choice_bool':  # 0.5, - 0.5 for choices left, right.
      bp_values[p] = choices.astype(bool)
    elif p == 'choice_stim':  # 0.5, - 0.5 for choices left, right.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (choices.astype(bool) & stim_bool) - 0.5
    elif p == 'choice_nonstim':  # 0.5, - 0.5 for choices left, right.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (choices.astype(bool) & ~stim_bool) - 0.5
    elif p == 'outcome':  # 0.5 , -0.5 for  rewarded , not rewarded.
      bp_values[p] = (outcomes == choices) - 0.5
    elif p == 'prev_outcome':  # 0.5 , -0.5 for  rewarded , not rewarded. (not interacting with choice)
      bp_values[p] = choices - 0.5
    elif p == 'outcome_single':  # 0.5 , -0.5 for  rewarded , not rewarded.
      bp_values[p] = outcomes - 0.5
    elif p == 'outcome_stim':  # 0.5 , -0.5 for  rewarded , not rewarded.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'outcome_nonstim':  # 0.5 , -0.5 for  rewarded , not rewarded.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'stim':
      bp_values[p] = stim.astype(float) - 0.5
    elif p == 'prev_stim': # as the predictor on top but changed name for consistency
      bp_values[p] = stim.astype(float) - 0.5
    elif p == 'stim_bool':
      bp_values[p] = stim.astype(bool)
    elif p == 'choice_x_stim':
      bp_values[p] = (choices == stim) - 0.5
    elif p == 'trans_x_stim':
      bp_values[p] = ((transitions_CR == stim) == choices) - 0.5
    elif p == 'out_x_stim':
      bp_values[p] = ((outcomes == stim) == choices) - 0.5
    elif p == 'trCR_x_out_x_stim':
      bp_values[p] = ((transition_CR_x_outcome == stim) == choices) - 0.5
    elif p == 'forced_choice':
      bp_values[p] = (forced_choice_trials == choices) - 0.5
    elif p == 'stay':  # 0.5 stay, -0.5 switch
      bp_values[p] = list(np.hstack(([0], (choices[1:] == choices[:-1]).astype(int) - 0.5)))
    elif p == 'trans_CR':  # 0.5, -0.5 for common, rare transitions.
      bp_values[p] = ((transitions_CR) == choices) - 0.5
    elif p == 'prev_trans_CR':  # 0.5, -0.5 for common, rare transitions. (not interacting with choice)
      bp_values[p] = transitions_CR - 0.5
    elif p == 'trans_CR_single':  # 0.5, -0.5 for common, rare transitions.
      bp_values[p] = transitions_CR - 0.5
    elif p == 'trans_CR_stim':  # 0.5, -0.5 for common, rare transitions.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (transitions_CR.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'trans_CR_nonstim':  # 0.5, -0.5 for common, rare transitions.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (transitions_CR.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'trCR_x_out':  # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
      bp_values[p] = (transition_CR_x_outcome == choices) - 0.5
    elif p == 'trCR_x_out_single':  # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
      bp_values[p] = transition_CR_x_outcome - 0.5
    elif p == 'trans_x_out':
      bp_values[p] = ((outcomes == choices) - 0.5)*(((transitions_CR) == choices) - 0.5)
    elif p == 'trCR_x_out_stim':  # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (transition_CR_x_outcome.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'trCR_x_out_nonstim':  # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (transition_CR_x_outcome.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'rew_com':  # Rewarded common transition predicts repeating choice.
      bp_values[p] = (outcomes.astype(bool) & transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'rew_rare':  # Rewarded rare transition predicts repeating choice.
      bp_values[p] = (outcomes.astype(bool) & ~transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'non_com':  # Non-rewarded common transition predicts repeating choice.
      bp_values[p] = (~outcomes.astype(bool) & transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'non_rare':  # Non-Rewarded rare transition predicts repeating choice.
      bp_values[p] = (~outcomes.astype(bool) & ~transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'rew_com_stim':  # Rewarded common transition stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & transitions_CR.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'rew_rare_stim':  # Rewarded rare transition stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & ~transitions_CR.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'non_com_stim':  # Non-rewarded common transition stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~outcomes.astype(bool) & transitions_CR.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'non_rare_stim':  # Non-Rewarded rare transition stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~outcomes.astype(bool) & ~transitions_CR.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'rew_com_nonstim':  # Rewarded common transition non-stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & transitions_CR.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'rew_rare_nonstim':  # Rewarded rare transition non-stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & ~transitions_CR.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'non_com_nonstim':  # Non-rewarded common transition non-stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~outcomes.astype(bool) & transitions_CR.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'non_rare_nonstim':  # Non-Rewarded rare transition non-stimulated predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~outcomes.astype(bool) & ~transitions_CR.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'stim_com': # Stim common transitions predicts repeating choice
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (stim_bool & transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'stim_rare':  # Stim rare transition predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (stim_bool & ~transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'nonstim_com':  # Non-stim common transition predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~stim_bool & transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'nonstim_rare':  # Non-stim rare transition predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~stim_bool & ~transitions_CR.astype(bool)) * (choices - 0.5)
    elif p == 'rew_stim':  # Rewarded stimulated trials predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'rew_nonstim':  # Rewarded non-stim trials predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (outcomes.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'non_stim':  # Non-rewarded stim predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~outcomes.astype(bool) & stim_bool) * (choices - 0.5)
    elif p == 'non_nonstim':  # Non-Rewarded non-stim trials predicts repeating choice.
      stim_bool = stim.astype(bool) if stim[0] is not np.nan else stim
      bp_values[p] = (~outcomes.astype(bool) & ~stim_bool) * (choices - 0.5)
    elif p == 'cum_rew': # cumulative reward
      bp_values[p] = list(np.cumsum(choices))
    elif p == 'ntrial': # trial number
      bp_values[p] = np.arange(len(choices))

  # Generate lagged predictors from base predictors.
  n_trials = len(choices)
  df = pd.DataFrame()
  for i, p in enumerate(predictors_lags):
    predictors = np.zeros(n_trials)
    if '-' in p:  # Get lag from predictor name.
      lag = int(p.split('-')[1])
      bp_name = p.split('-')[0]
      predictors[lag:] = bp_values[bp_name][:-lag]

    elif single_lag:
      lag = single_lag
      bp_name = p
      predictors[lag:] = bp_values[bp_name][:-lag]
    else:  # Use default lag, 0 - no lag.
      lag = 0
      bp_name = p
      predictors[lag:] = bp_values[bp_name][:]

    df[p] = predictors

  return df


def _session_regressors(session, base_predictors, lags, single_lag, selection_type='all', select_n='all', block_type='all', stay_pr=False, opto=True, regressor_type='str', subjects_virus=[]):
  '''
  return regressors of a single session
  '''

  df_session = _get_dataframe_session(session, base_predictors=base_predictors, stay_pr=stay_pr, opto=opto, regressor_type=regressor_type, lags=lags, subjects_virus=subjects_virus)
  if stay_pr is False:
    df_predictors = _get_dataframe_session_regressors(df_session, base_predictors, lags, single_lag)

    df_session = pd.concat([df_session, df_predictors], axis=1)

    # extra columns with same regressors as computed but as dummy variables or effect (-1/1)
    if 'stim' in base_predictors:
      df_session['stim_bool'] = np.where(df_session.stim == 0.5, True,
                                             np.where(df_session.stim == -0.5, False, False))
      df_session['stim_int'] = np.where(df_session.stim == 0.5, 1,
                                        np.where(df_session.stim == -0.5, 0, 0))
      df_session['stim_regres'] = np.where(df_session.stim == 0.5, 1,
                                           np.where(df_session.stim == -0.5, -1, 0))
      df_session['stim_str'] = np.where(df_session.stim == 0.5, 'yes',
                                        np.where(df_session.stim == -0.5, 'no', 'no'))
      if 'choice' in base_predictors:
        df_session['choice_x_stim_int'] = np.where(df_session.choice_x_stim == 0.5, 1,
                                                 np.where(df_session.choice_x_stim == -0.5, 0, 0))
        df_session['choice_x_stim_int11'] = np.where(
          (df_session['stim'] == 0.5) & (df_session['choice'] == 0.5), 1, 0)
        df_session['choice_x_stim_regres'] = np.where(df_session.choice_x_stim == 0.5, 1,
                                                      np.where(df_session.choice_x_stim == -0.5, -1, 0))
        df_session['choice_x_stim_regress_1_1'] = np.where(
          (df_session['stim'] == 0.5) & (df_session['choice'] == 0.5), -1, 1)

      if 'outcome' in base_predictors:
        df_session['out_x_stim_regres'] = np.where(df_session.out_x_stim == 0.5, 1,
                                                      np.where(df_session.out_x_stim == -0.5, -1, 0))
      if 'trans_CR' in base_predictors:
        df_session['trans_x_stim_regres'] = np.where(df_session.trans_x_stim == 0.5, 1,
                                                   np.where(df_session.trans_x_stim == -0.5, -1, 0))
      if 'trCR_x_out' in base_predictors:
        df_session['trCR_x_out_x_stim_regres'] = np.where(df_session.trCR_x_out_x_stim == 0.5, 1,
                                                   np.where(df_session.trCR_x_out_x_stim == -0.5, -1, 0))

    if 'choice' in base_predictors:
      df_session['choice_bool'] = np.where(df_session.choice == 0.5, True,
                                               np.where(df_session.choice == -0.5, False, False))
      df_session['choice_int'] = np.where(df_session.choice == 0.5, 1,
                                          np.where(df_session.choice == -0.5, 0, 0))
      df_session['choice_regres'] = np.where(df_session.choice == 0.5, 1,
                                             np.where(df_session.choice == -0.5, -1, 0))
      df_session['choice_str'] = np.where(df_session.choice == 0.5, 'yes',
                                          np.where(df_session.choice == -0.5, 'no', 'no'))
    if 'outcome' in base_predictors:
      df_session['outcome_bool'] = np.where(df_session.outcome == 0.5, True,
                                                np.where(df_session.outcome == -0.5, False, False))
      df_session['outcome_int'] = np.where(df_session.outcome == 0.5, 1,
                                           np.where(df_session.outcome == -0.5, 0, 0))
      df_session['outcome_regres'] = np.where(df_session.outcome == 0.5, 1,
                                              np.where(df_session.outcome == -0.5, -1, 0))
      df_session['outcome_single_regres'] = np.where(df_session.outcome_single == 0.5, 1,
                                              np.where(df_session.outcome_single == -0.5, -1, 0))
    if 'trans_CR' in base_predictors:
      df_session['trans_CR_bool'] = np.where(df_session.trans_CR == 0.5, True,
                                                 np.where(df_session.trans_CR == -0.5, False, False))
      df_session['trans_CR_int'] = np.where(df_session.trans_CR == 0.5, 1,
                                            np.where(df_session.trans_CR == -0.5, 0, 0))
      df_session['trans_CR_regres'] = np.where(df_session.trans_CR == 0.5, 1,
                                               np.where(df_session.trans_CR == -0.5, -1, 0))
      df_session['trans_CR_single_regres'] = np.where(df_session.trans_CR_single == 0.5, 1,
                                               np.where(df_session.trans_CR_single == -0.5, -1, 0))
    if 'trCR_x_out' in base_predictors:
      df_session['trCR_x_out_bool'] = np.where(df_session.trCR_x_out == 0.5, True,
                                                   np.where(df_session.trCR_x_out == -0.5, False, False))
      df_session['trCR_x_out_int'] = np.where(df_session.trCR_x_out == 0.5, 1,
                                              np.where(df_session.trCR_x_out == -0.5, 0, 0))
      df_session['trCR_x_out_regres'] = np.where(df_session.trCR_x_out == 0.5, 1,
                                                 np.where(df_session.trCR_x_out == -0.5, -1, 0))
      df_session['trCR_x_out_single_regres'] = np.where(df_session.trCR_x_out_single == 0.5, 1,
                                                 np.where(df_session.trCR_x_out_single == -0.5, -1, 0))
    if 'correct' in base_predictors:
      df_session['correct_int'] = np.where(df_session.trCR_x_out == 0.5, 1,
                                               np.where(df_session.trCR_x_out == -0.5, 0, -1))
      df_session['correct_regres'] = np.where(df_session.correct == 0.5, 1,
                                                  np.where(df_session.correct == -0.5, -1, 0))
      df_session['correct_str'] = np.where(df_session.stim == 0.5, 'correct',
                                           np.where(df_session.stim == -0.5, 'incorrect', 'neutral'))

    #extra columns that could be used to mask and select trials
    df_session['positions_mask'] = session.select_trials(selection_type=selection_type, select_n=select_n, block_type=block_type)
  else:
    df_session['positions_mask'] = session.select_trials(selection_type=selection_type, select_n=select_n,
                                                         block_type=block_type)[:-1]

  return df_session

def import_regressors(sessions, base_predictors, lags, single_lag, selection_type='all', select_n='all', block_type='all', stay_pr=False, opto=True, regressor_type='str', subjects_virus=[]):
  list_sessions = pp.map(partial(_session_regressors, base_predictors=base_predictors, lags=lags, single_lag=single_lag,
                                                 selection_type=selection_type, select_n=select_n,
                                                 block_type=block_type, stay_pr=stay_pr, opto=opto,
                                 regressor_type=regressor_type, subjects_virus=subjects_virus), sessions)

  df_sessions = pd.concat(list_sessions)

  return df_sessions



#%% rebuttal
#for lagged regression predicting mean dopamine activity at a range (e.g. at ss step)

def _get_dataframe_session_predictors_pho(session, pho_activity, base_predictors, lags=[]):

  choices = session.trial_data['choices'].astype(bool)
  transitions_AB = session.trial_data['transitions'].astype(bool)
  second_steps = session.trial_data['second_steps'].astype(bool)
  second_steps_l1 = _lag(second_steps, 1)
  outcomes = session.trial_data['outcomes'].astype(bool)
  reward_l1 = _lag(outcomes, 1)

  stay = choices[1:] == choices[:-1]

  rew_state = session.blocks['trial_rew_state']
  trans_state = session.blocks['trial_trans_state']  # Trial by trial state of the tranistion matrix (A vs B)
  transitions_CR = transitions_AB == trans_state  # Trial by trial common or rare transitions (True: common; False: rare)
  transition_CR_x_outcome = transitions_CR == outcomes  # True: common and rewarded / rare and unrewarded
                                                        # False: common and unrewarded / rare and rewarded
  forced_choice_trials = ~session.trial_data['free_choice']
  forced_choice_next_trial = forced_choice_trials[1:]
  subject = np.repeat(session.subject_ID, len(choices))
  subject_str = np.asarray(['m{}'.format(sub) for sub in subject])
  try:
    date = np.repeat(session.datetime_string.split()[0], len(choices))
  except AttributeError:
    date = np.repeat(['no_date'], len(choices))

  lags_shift = np.arange(np.abs(lags))
  d2 = {}
  if 'second_step_same_rew' in base_predictors: # 1 if previous ss same as current and previously rewarded
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_same_rew'] = np.asarray([1 if (ss == 1 and r1 == 1) else 0
                               for ss, r1 in zip(same_ss, reward_l1)])

  if 'second_step_same_nonrew' in base_predictors: # 1 if previous ss same as current and previously nonrewarded
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_same_nonrew'] = np.asarray([1 if (ss == 1 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_l1)])

  if 'second_step_diff_rew' in base_predictors: # 1 if previous ss different as current and previously rewarded
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_diff_rew'] = np.asarray([1 if (ss == 0 and r1 == 1) else 0
                               for ss, r1 in zip(same_ss, reward_l1)])

  if 'second_step_diff_nonrew' in base_predictors: # 1 if previous ss different as current and previously nonrewarded
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_diff_nonrew'] = np.asarray([1 if (ss == 0 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_l1)])

  if 'second_step_update_same' in base_predictors:
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_update_same'] = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_l1)])

  if 'second_step_update_diff' in base_predictors:
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_update_diff'] = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_l1)])

  if 'second_step_same_rew_x_trans' in base_predictors: # 0.5 common, -0.5 rare if previous ss same as current and previously rewarded
    same_ss = (second_steps == second_steps_l1) * 1
    transitions_CR_lag = _lag(transitions_CR, 1)
    d2['second_step_same_rew_x_trans'] = np.asarray([0.5 if (ss == 1 and r1 == 1 and t == 1)
                                                               else -0.5 if (ss == 1 and r1 == 1 and t == 0) else 0
                               for ss, r1, t in zip(same_ss, reward_l1, transitions_CR)])

  if 'second_step_same_nonrew_x_trans' in base_predictors: # 1 if previous ss same as current and previously nonrewarded
    same_ss = (second_steps == second_steps_l1) * 1
    transitions_CR_lag = _lag(transitions_CR, 1)
    d2['second_step_same_nonrew_x_trans'] = np.asarray([0.5 if (ss == 1 and r1 == 0 and t == 1)
                                                               else -0.5 if (ss == 1 and r1 == 0 and t == 0) else 0
                               for ss, r1, t in zip(same_ss, reward_l1, transitions_CR)])

  if 'second_step_diff_rew_x_trans' in base_predictors: # 1 if previous ss different as current and previously rewarded
    same_ss = (second_steps == second_steps_l1) * 1
    transitions_CR_lag = _lag(transitions_CR, 1)
    d2['second_step_diff_rew_x_trans'] = np.asarray([0.5 if (ss == 0 and r1 == 1 and t == 1)
                                                        else -0.5 if (ss == 0 and r1 == 1 and t == 0) else 0
                                                        for ss, r1, t in zip(same_ss, reward_l1, transitions_CR)])

  if 'second_step_diff_nonrew_x_trans' in base_predictors: # 1 if previous ss different as current and previously nonrewarded
    same_ss = (second_steps == second_steps_l1) * 1
    transitions_CR_lag = _lag(transitions_CR, 1)
    d2['second_step_diff_nonrew_x_trans'] = np.asarray([0.5 if (ss == 0 and r1 == 0 and t == 1)
                                                        else -0.5 if (ss == 0 and r1 == 0 and t == 0) else 0
                                                        for ss, r1, t in zip(same_ss, reward_l1, transitions_CR)])

  d1 = {'subject': subject,
        'subject_str': subject_str,
        'date': date,
        'pho': pho_activity,
        }
  d = {**d1, **d2}

  df = pd.DataFrame(data=d)

  if lags:
    lags_shift = np.arange(np.abs(lags))
    for bp in base_predictors:
      predictors = np.zeros(len(df['pho']))
      for lg in lags_shift:
        if lg == 0:
          df['{}_{}'.format(bp, lg)] = df[bp]
        else:
          df['{}_{}'.format(bp, lg)] = _lag(df[bp], lg)

  return df

def _get_dataframe_session_predictors_pho_2(session, pho_activity, base_predictors, lags=[], no_lag_predictors=[],
                                            hemisphere=[]):

  choices = session.trial_data['choices'].astype(bool)
  transitions_AB = session.trial_data['transitions'].astype(bool)
  second_steps = session.trial_data['second_steps'].astype(bool)
  choices_l1 = _lag(choices, 1)
  second_steps_l1 = _lag(second_steps, 1)
  outcomes = session.trial_data['outcomes'].astype(bool)
  reward_l1 = _lag(outcomes, 1)

  stay = choices[1:] == choices[:-1]

  rew_state = session.blocks['trial_rew_state']
  trans_state = session.blocks['trial_trans_state']  # Trial by trial state of the tranistion matrix (A vs B)
  transitions_CR = transitions_AB == trans_state  # Trial by trial common or rare transitions (True: common; False: rare)
  transition_CR_x_outcome = transitions_CR == outcomes  # True: common and rewarded / rare and unrewarded
                                                        # False: common and unrewarded / rare and rewarded
  forced_choice_trials = ~session.trial_data['free_choice']
  forced_choice_next_trial = forced_choice_trials[1:]
  subject = np.repeat(session.subject_ID, len(choices))
  subject_str = np.asarray(['m{}'.format(sub) for sub in subject])
  try:
    date = np.repeat(session.datetime_string.split()[0], len(choices))
  except AttributeError:
    date = np.repeat(['no_date'], len(choices))

  lags_shift = np.arange(np.abs(lags))
  d2 = {}
  if 'second_step_same_rew' in base_predictors: # 1 if previous ss same as current and previously rewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      d2['second_step_same_rew_{}'.format(lg)] = np.asarray([1 if (ss == 1 and r1 == 1) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])
  if 'second_step_same_nonrew' in base_predictors: # 1 if previous ss same as current and previously nonrewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      d2['second_step_same_nonrew_{}'.format(lg)] = np.asarray([1 if (ss == 1 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])
  if 'second_step_diff_rew' in base_predictors: # 1 if previous ss different as current and previously rewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      d2['second_step_diff_rew_{}'.format(lg)] = np.asarray([1 if (ss == 0 and r1 == 1) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])
  if 'second_step_diff_nonrew' in base_predictors: # 1 if previous ss different as current and previously nonrewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      d2['second_step_diff_nonrew_{}'.format(lg)] = np.asarray([1 if (ss == 0 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])

  if 'second_step_update_same' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      d2['second_step_update_same_{}'.format(lg)] = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])
  if 'second_step_update_diff' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      d2['second_step_update_diff_{}'.format(lg)] = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])

  if 'second_step_update_same_x_currtrans' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      ss_update_same = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])

      d2['second_step_update_same_x_currtrans_{}'.format(lg)] = np.asarray([sus * -1 if t == 0 else sus
                                                                            for sus, t in zip(ss_update_same, transitions_CR)])
  if 'second_step_update_diff_x_currtrans' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      ss_update_diff = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                               for ss, r1 in zip(same_ss, reward_lag)])

      d2['second_step_update_diff_x_currtrans_{}'.format(lg)] = np.asarray([sud * -1 if t == 0 else sud
                                                                            for sud, t in zip(ss_update_diff, transitions_CR)])

  if 'second_step_update_same_x_prevtrans' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg + 1)
      ss_update_same = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                                   for ss, r1 in zip(same_ss, reward_lag)])
      transitions_CR_lag = _lag(transitions_CR, lg + 1)
      d2['second_step_update_same_x_prevtrans_{}'.format(lg)] = np.asarray([sus * -1 if t == 0 else sus
                                                                            for sus, t in
                                                                            zip(ss_update_same, transitions_CR_lag)])
  if 'second_step_update_diff_x_prevtrans' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg + 1)
      ss_update_diff = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                                   for ss, r1 in zip(same_ss, reward_lag)])
      transitions_CR_lag = _lag(transitions_CR, lg + 1)
      d2['second_step_update_diff_x_prevtrans_{}'.format(lg)] = np.asarray([sud * -1 if t == 0 else sud
                                                                            for sud, t in
                                                                            zip(ss_update_diff, transitions_CR_lag)])

  if 'second_step_same_rew_x_trans' in base_predictors: # 0.5 common, -0.5 rare if previous ss same as current and previously rewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg+1)
      transitions_CR_lag = _lag(transitions_CR, lg+1)
      d2['second_step_same_rew_x_trans_{}'.format(lg)] = np.asarray([0.5 if (ss == 1 and r1 == 1 and t == 1)
                                                               else -0.5 if (ss == 1 and r1 == 1 and t == 0) else 0
                               for ss, r1, t in zip(same_ss, reward_lag, transitions_CR_lag)])
  if 'second_step_same_nonrew_x_trans' in base_predictors: # 1 if previous ss same as current and previously nonrewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg + 1)
      transitions_CR_lag = _lag(transitions_CR, lg + 1)
      d2['second_step_same_nonrew_x_trans_{}'.format(lg)] = np.asarray([0.5 if (ss == 1 and r1 == 0 and t == 1)
                                                                  else -0.5 if (ss == 1 and r1 == 0 and t == 0) else 0
                                        for ss, r1, t in zip(same_ss, reward_lag, transitions_CR_lag)])
  if 'second_step_diff_rew_x_trans' in base_predictors: # 1 if previous ss different as current and previously rewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg + 1)
      transitions_CR_lag = _lag(transitions_CR, lg + 1)
      d2['second_step_diff_rew_x_trans_{}'.format(lg)] = np.asarray([0.5 if (ss == 0 and r1 == 1 and t == 1)
                                                               else -0.5 if (ss == 0 and r1 == 1 and t == 0) else 0
                                                          for ss, r1, t in zip(same_ss, reward_lag, transitions_CR_lag)])
  if 'second_step_diff_nonrew_x_trans' in base_predictors: # 1 if previous ss different as current and previously nonrewarded
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      same_ss = (second_steps == second_steps_lag) * 1
      reward_lag = _lag(outcomes, lg + 1)
      transitions_CR_lag = _lag(transitions_CR, lg + 1)
      d2['second_step_diff_nonrew_x_trans_{}'.format(lg)] = np.asarray([0.5 if (ss == 0 and r1 == 0 and t == 1)
                                                                  else -0.5 if (ss == 0 and r1 == 0 and t == 0) else 0
                                                        for ss, r1, t in zip(same_ss, reward_lag, transitions_CR_lag)])

  if 'second_step_same_diff' in base_predictors: # 1 if previous ss same as current
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg+1)
      same_ss = (second_steps == second_steps_lag) * 1
      d2['second_step_same_diff_{}'.format(lg)] = np.asarray([1 if ss == 1 else 0
                               for ss in same_ss])

  if 'same_choice_x_rew' in base_predictors: #same as model_free_update
    for lg in lags_shift:
      choices_lag = _lag(choices, lg + 1)
      same_ch = (choices == choices_lag) * 1
      reward_lag = _lag(outcomes, lg + 1)
      d2['same_choice_x_rew_{}'.format(lg)] = np.asarray(
        [0.5 if (x == 1 and r1 == 1) else -0.5 if (x == 1 and r1 == 0) else 0 for x, r1 in zip(same_ch, reward_lag)])

  if 'diff_choice_x_rew' in base_predictors:
    for lg in lags_shift:
      choices_lag = _lag(choices, lg + 1)
      same_ch = (choices == choices_lag) * 1
      reward_lag = _lag(outcomes, lg + 1)
      d2['diff_choice_x_rew_{}'.format(lg)] = np.asarray(
        [0.5 if (x == 0 and r1 == 1) else -0.5 if (x == 0 and r1 == 0) else 0 for x, r1 in zip(same_ch, reward_lag)])

  if 'choice_to_same_ss_x_rew' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      choices_to_ssl1 = (~(second_steps_lag ^ trans_state).astype(bool)).astype(int)
      choice_to_same_ss = (choices_to_ssl1 == choices) * 1
      reward_lag = _lag(outcomes, lg + 1)

      d2['choice_to_same_ss_x_rew_{}'.format(lg)] = np.asarray([0.5 if (x == 1 and r1 == 1) else -0.5 if (x == 1 and r1 == 0) else 0
                                                                for x, r1 in zip(choice_to_same_ss, reward_lag)])
  if 'choice_to_diff_ss_x_rew' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      choices_to_ssl1 = (~(second_steps_lag ^ trans_state).astype(bool)).astype(int)
      choice_to_same_ss = (choices_to_ssl1 == choices) * 1
      reward_lag = _lag(outcomes, lg + 1)

      d2['choice_to_diff_ss_x_rew_{}'.format(lg)] = np.asarray([0.5 if (x == 0 and r1 == 1) else -0.5 if (x == 0 and r1 == 0) else 0
                                                                for x, r1 in zip(choice_to_same_ss, reward_lag)])
  if 'model_based_update' in base_predictors:
    for lg in lags_shift:
      second_steps_lag = _lag(second_steps, lg + 1)
      choices_to_ssl1 = (~(second_steps_lag ^ trans_state).astype(bool)).astype(int)
      reward_lag = _lag(outcomes, lg + 1)
      d2['model_based_update_{}'.format(lg)] = np.asarray([0.5 if x == 1 else -0.5 for x in ((choices_to_ssl1 == choices) == reward_lag) * 1])

  if 'reward' in base_predictors:
    for lg in lags_shift:
      reward_lag = _lag(outcomes, lg + 1)
      d2['reward_{}'.format(lg)] = np.asarray([1 if rl == 1 else 0 for rl in reward_lag])

  # No lag predictors

  if 'reward' in no_lag_predictors:
    # need to check in the lagged if this is changed for 0.5 if the same results
    d2['reward'] = np.asarray([0.5 if o == 1 else -0.5 for o in (outcomes * 1)])

  if 'reward_1' in no_lag_predictors:
    d2['reward_1'] = np.asarray([0.5 if o == 1 else -0.5 for o in (reward_l1 * 1)])

  if 'transition_CR' in no_lag_predictors:
    d2['transition_CR'] = transitions_CR.astype(int)

  if 'reward_rate' in no_lag_predictors:
    moving_reward_rate = exp_mov_ave(tau=8, init_value=0.5)
    moving_reward_average_session = []
    for x in reward_l1:
      moving_reward_rate.update(x)
      moving_reward_average_session.append(moving_reward_rate.value)
    d2['reward_rate'] = np.asarray(moving_reward_average_session)

  if 'short_reward_rate' in no_lag_predictors:
    moving_reward_rate = exp_mov_ave(tau=5, init_value=0.5)
    moving_reward_average_session = []
    for x in reward_l1:
      moving_reward_rate.update(x)
      moving_reward_average_session.append(moving_reward_rate.value)
    d2['short_reward_rate'] = np.asarray(moving_reward_average_session)

  if 'long_reward_rate' in no_lag_predictors:
    moving_reward_rate = exp_mov_ave(tau=15, init_value=0.5)
    moving_reward_average_session = []
    for x in reward_l1:
      moving_reward_rate.update(x)
      moving_reward_average_session.append(moving_reward_rate.value)
    d2['long_reward_rate'] = np.asarray(moving_reward_average_session)

  if 'good_ss_rate' in no_lag_predictors:
    good_ss = np.asarray([0.5 if (ss == rs and rs != 2) else -0.5 if (ss != rs and rs != 2) else 0
                               for ss, rs in zip(second_steps, rew_state)])
    moving_rate = exp_mov_ave(tau=8, init_value=0.5)
    moving_average_session = []
    for x in good_ss:
      moving_rate.update(x)
      moving_average_session.append(moving_rate.value)
    d2['good_ss_rate'] = np.asarray(moving_average_session)

  if 'second_step_update_same' in no_lag_predictors:
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_update_same'] = np.asarray([0.5 if (ss == 1 and r1 == 1) else -0.5 if (ss == 1 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])

  if 'second_step_update_diff' in no_lag_predictors:
    same_ss = (second_steps == second_steps_l1) * 1
    d2['second_step_update_diff'] = np.asarray([0.5 if (ss == 0 and r1 == 1) else -0.5 if (ss == 0 and r1 == 0) else 0
                                 for ss, r1 in zip(same_ss, reward_l1)])

  if 'model_free_update' in no_lag_predictors:
    same_ch = (choices == choices_l1) * 1
    d2['model_free_update'] = np.asarray([0.5 if (x == 1 and r1 == 1) else -0.5 if (x == 1 and r1 == 0) else 0 for x, r1
                                          in zip(same_ch, reward_l1)])

  if 'model_based_update' in no_lag_predictors:
    choices_to_ssl1 = (~(second_steps_l1 ^ trans_state).astype(bool)).astype(int)
    d2['model_based_update'] = np.asarray([0.5 if x == 1 else -0.5 for x in ((choices_to_ssl1 == choices) == reward_l1) * 1])

  if 'good_ss' in no_lag_predictors:
    d2['good_ss'] = np.asarray([0.5 if (ss == rs and rs != 2) else -0.5 if (ss != rs and rs != 2) else 0
                               for ss, rs in zip(second_steps, rew_state)])

  if 'good_ss_l1' in no_lag_predictors:
    d2['good_ss_l1'] = _lag(np.asarray([0.5 if (ss == rs and rs != 2) else -0.5 if (ss != rs and rs != 2) else 0
                               for ss, rs in zip(second_steps, rew_state)]), 1)

  if 'contralateral_choice' in no_lag_predictors:
    hemisphere_param = [1 if hemisphere == 'L' else -1][0]
    d2['contralateral_choice'] = np.asarray([-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param for c in (choices * 1)])

  if 'up_down' in no_lag_predictors:
    d2['up_down'] = np.asarray([0.5 if ss == 1 else -0.5 for ss in (second_steps * 1)])

  if 'correct_choice' in no_lag_predictors:
    correct = ((~choices.astype(bool)).astype(int) == (trans_state ^ rew_state))
    d2['correct_choice'] = np.asarray([0 if rs ==2 else 0.5 if c == True else -0.5 for c, rs in zip(correct, rew_state)])

  if 'repeat_choice' in no_lag_predictors:
    same_ch = (choices == choices_l1) * 1
    d2['repeat_choice'] = np.asarray([0.5 if c == 1 else -0.5 for c in same_ch])

  if 'forced_choice_single' in no_lag_predictors:
    d2['forced_choice_single'] = np.asarray([0.5 if x ==1 else -0.5 for x in forced_choice_trials * 1])

  if 'common_rare' in no_lag_predictors:
    d2['common_rare'] = np.asarray([0.5 if t == True else -0.5 for t in transitions_CR])

  if 'ss_rew_rate' in no_lag_predictors:
    mov_rew_rate_up = exp_mov_ave(tau=8, init_value=0.5)
    mov_rew_rate_down = exp_mov_ave(tau=8, init_value=0.5)
    mov_rew_rate_up_session = []
    mov_rew_rate_down_session = []
    for r, ss in zip(outcomes * 1, second_steps * 1):
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

    d2['ss_rew_rate'] = np.asarray([rate[ss] for rate, ss in zip(all_rate, second_steps*1)])

  if 'ss_rew_rate_l1' in no_lag_predictors:
      mov_rew_rate_up = exp_mov_ave(tau=8, init_value=0.5)
      mov_rew_rate_down = exp_mov_ave(tau=8, init_value=0.5)
      mov_rew_rate_up_session = []
      mov_rew_rate_down_session = []
      for r, ss in zip(outcomes * 1, second_steps * 1):
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

      d2['ss_rew_rate_l1'] = np.asarray([rate[ss] for rate, ss in zip(all_rate, second_steps_l1*1)])

  if 'cumsum_block' in no_lag_predictors:
    cumsum_block = pd.DataFrame(pd.DataFrame(session.blocks['trial_rew_state'],
                                             columns=['trial_rew_state'])['trial_rew_state'].diff().ne(0).cumsum().values,
                                columns=['cumsum']).groupby('cumsum').cumcount().values
    d2['cumsum_block'] = (cumsum_block - np.min(cumsum_block))/(np.max(cumsum_block) - np.min(cumsum_block))

  if 'start_end_block' in no_lag_predictors:
    d2['start_end_block'] = session.select_trials(selection_type='end', select_n=15, block_type='all') * -0.5 + \
                     session.select_trials(selection_type='start', select_n=15, block_type='all') * 0.5

  if 'same_diff_rate' in no_lag_predictors:
    moving_ss_rate = exp_mov_ave(tau=8, init_value=0.5)
    moving_ss_average_session = []
    same_ss = (second_steps == second_steps_l1) * 1
    for x in same_ss:
      moving_ss_rate.update(x)
      moving_ss_average_session.append(moving_ss_rate.value)
    d2['same_diff_rate'] = np.asarray(moving_ss_average_session)

  if 'n_same' in no_lag_predictors:
    same_ss = (second_steps == second_steps_l1) * 1
    df_same_ss = pd.DataFrame(same_ss.astype(bool), columns=['bool'])
    n_same = df_same_ss.groupby(df_same_ss['bool'].astype(int).diff().ne(0).cumsum())['bool'].cumsum().values
    d2['n_same'] = (n_same - np.min(n_same))/(np.max(n_same) - np.min(n_same))

  if 'n_diff' in no_lag_predictors:
    same_ss = (second_steps == second_steps_l1) * 1
    df_diff_ss = pd.DataFrame(~same_ss.astype(bool), columns=['bool'])
    n_diff = df_diff_ss.groupby(df_diff_ss['bool'].astype(int).diff().ne(0).cumsum())['bool'].cumsum().values
    d2['n_diff'] = (n_diff - np.min(n_diff))/(np.max(n_diff) - np.min(n_diff))


  d1 = {'subject': subject,
        'subject_str': subject_str,
        'date': date,
        'pho': pho_activity,
        }
  d = {**d1, **d2}

  df = pd.DataFrame(data=d)

  return df

def import_regressors_pho(sessions, pho_activity_all_sessions, base_predictors, lags, import_2=False, no_lag_predictors=[], hemisphere=[]):
  if import_2:
    list_sessions = pp.starmap(_get_dataframe_session_predictors_pho_2, zip(sessions, pho_activity_all_sessions,
                                                                          [base_predictors] * len(sessions),
                                                                          [lags] * len(sessions),
                                                                            [no_lag_predictors] * len(sessions),
                                                                            [hemisphere] * len(sessions)))
  else:
    list_sessions = pp.starmap(_get_dataframe_session_predictors_pho, zip(sessions, pho_activity_all_sessions,
                                                                          [base_predictors]*len(sessions),
                                                                          [lags]*len(sessions)))
  df_sessions = pd.concat(list_sessions)

  return df_sessions

