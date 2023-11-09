# ---------------------------------------------------------------------------------------------------------------
# Functions to import data and generate the main figures from the manuscript (Blanco-Pozo, Akam & Walton, 2023)
# Marta Blanco-Pozo, 2023
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------------
import sys, os
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from code import plot_behaviour as pl, saving as sv, simulation as sim, \
  import_photometry_data as pi, model_fitting as mf, plot_models as pm, import_behaviour_data as di, \
  plot_photometry as plp, parallel_processing as pp

from RL_agents_two_step import mf, mb, latent_state, mf_forget_diffa, mb_forget_0_diffa, latent_state_rewasym

#dir_folder_variables = './data_variables/'

# ---------------------------------------------------------------------------------------------------------------
# 1- Import manuscript data
# ---------------------------------------------------------------------------------------------------------------
def preprocess_data(dir_folder_session_1, dir_folder_pho_1, dir_folder_session_2, dir_folder_pho_2, **kwargs):
  '''
  Preprocess data - Import and/or save behaviour and photometry data as in the manuscript
  :param dir_folder_session_1: folder path of cohort 1 behavioural files
  :param dir_folder_pho_1: folder path of cohort 1 photometry files
  :param dir_folder_session_2: folder path of cohort 2 behavioural files
  :param dir_folder_pho_2: folder path of cohort 2 photometry files
  :param kwargs:
    save_folder: if folder path is provided, it will save the variables in that directory
    return_variables: if True, it will return a dictionary with the data:
      {sessions: list with all the sessions,
      photometry: list with all the photometry data for each session,
      sample_times_pho: list of the times for each photometry timepoint in the 'photometry clock' (pyPhotometry),
      sample_times_pyc: list of the times for each photometry timepoint in the 'behavioural clock' (pyControl),
      corrected_photometry: list of the (bleached and motion) corrected photometry signals
      photometry_simplified_info: a simplified version of the photometry variable
  '''
  save_folder = kwargs.pop('save_folder', [])
  return_variables = kwargs.pop('return_variables', False)

  # cohort_1
  mouse_1 = [20, 21, 24, 25, 26, 27, 28]
  day_1 = ['2019-04-28', '2019-04-29', '2019-04-30', '2019-05-01', '2019-05-02', '2019-05-03', '2019-05-04',
           '2019-05-05',
           '2019-05-06', '2019-05-07', '2019-05-08', '2019-05-09', '2019-05-10', '2019-05-11', '2019-05-12',
           '2019-05-13', '2019-05-14', '2019-05-15', '2019-05-17', '2019-05-18', '2019-05-19', '2019-05-20',
           '2019-05-21',
           '2019-05-22', '2019-05-23', '2019-05-28', '2019-05-29', '2019-05-30']
  # cohort_2
  mouse_2 = [48, 49, 51, 52, 53, 54, 55, 56, 57, 61, 62]
  day_2 = ['2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-23', '2019-11-24',
           '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', '2019-12-01',
           '2019-12-02', '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-08', '2019-12-09', '2019-12-10',
           '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17',
           '2019-12-18', '2019-12-19', '2019-12-20']

  all_sessions_path_1, all_photometry_path_1 = pi.import_sessions_photometry_path(
    dir_folder_session_1, dir_folder_pho_1, start_str='m', sessions_format='m{id}-{datetime}.txt',
    photometry_format='m{id}_{region}_{hemisphere}-{datetime}.ppd', mouse=mouse_1, day=day_1, region=[], hemisphere=[])

  all_sessions_path_2, all_photometry_path_2 = pi.import_sessions_photometry_path(
    dir_folder_session_2, dir_folder_pho_2, start_str='m', sessions_format='m{id}-{datetime}.txt',
    photometry_format='m{id}_{region}_{hemisphere}-{datetime}.ppd', mouse=mouse_2, day=day_2, region=[], hemisphere=[],
    exclusion=['m48-2019-11-18-112102.txt', 'm48_DMS_R-2019-11-18-112050.ppd',
               'm53-2019-11-29-093422.txt', 'm53_DMS_R-2019-11-29-093413.ppd'])

  sessions_1 = [di.Session(s_path) for s_path in all_sessions_path_1]
  pp.enable_multiprocessing()
  all_photo_data_1, all_sample_times_pho_1, all_sample_times_pyc_1, all_corrected_signal_1 = \
    zip(*pp.starmap(pi.sync_photometry_data,
                    zip(all_photometry_path_1, np.repeat('m{id}_{region}_{hemisphere}', len(all_photometry_path_1)),
                        sessions_1, np.repeat(5, len(all_photometry_path_1)),
                        np.repeat(0.001, len(all_photometry_path_1)))))
  pp.disable_multiprocessing()
  all_photo_data_info_1 = pi.session_information_from_photo_data(all_photo_data_1)

  pp.enable_multiprocessing()
  sessions_2 = [di.Session(s_path) for s_path in all_sessions_path_2]
  all_photo_data_2, all_sample_times_pho_2, all_sample_times_pyc_2, all_corrected_signal_2 = \
    zip(*pp.starmap(pi.sync_photometry_data,
                    zip(all_photometry_path_2, np.repeat('m{id}_{region}_{hemisphere}', len(all_photometry_path_2)),
                        sessions_2, np.repeat(5, len(all_photometry_path_2)),
                        np.repeat(0.001, len(all_photometry_path_2)))))
  pp.disable_multiprocessing()
  all_photo_data_info_2 = pi.session_information_from_photo_data(all_photo_data_2)

  sessions_together = sessions_1 + sessions_2
  all_photo_data_together = all_photo_data_1 + all_photo_data_2
  all_sample_times_pho_together = all_sample_times_pho_1 + all_sample_times_pho_2
  all_sample_times_pyc_together = all_sample_times_pyc_1 + all_sample_times_pyc_2
  all_corrected_signal_together = all_corrected_signal_1 + all_corrected_signal_2
  all_photo_data_info = all_photo_data_info_1 + all_photo_data_info_2

  if save_folder != []:
    sv.save_variable_joblib(sessions_together, 'sessions_together', save_folder)
    sv.save_variables_separately_joblib([all_photo_data_together, all_sample_times_pho_together, all_sample_times_pyc_together,
                                         all_corrected_signal_together, all_photo_data_info],
                                        ['all_photo_data_together', 'all_sample_times_pho_together', 'all_sample_times_pyc_together',
                                         'all_corrected_signal_together', 'all_photo_data_info'],
                                        save_folder)

  if return_variables:
    return {'sessions': sessions_together,
            'photometry': all_photo_data_together,
            'times_photometry': all_sample_times_pho_together,
            'times_behaviour': all_sample_times_pyc_together,
            'corrected_photometry': all_corrected_signal_together,
            'photometry_simplified_info': all_photo_data_info}

def preprocess_data_cohort(mice, day, dir_folder_sessions, dir_folder_photometry, **kwargs):
  '''
  Preprocess data - Import and/or save behaviour and photometry data (as the function above but user can define what they want to import)
  :param mice: list of mice to import. e.g. [48, 49, 51, 52, 53, 54, 55, 56, 57, 61, 62]
  :param day: list of days to import. e.g. ['2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-23', '2019-11-24']
  :param dir_folder_sessions: path of the folder containing the pyControl files
  :param dir_folder_photometry: path of the folder containing the pyPhotometry files
  :param kwargs:
    exclusion: list of pyControl and pyPhotometry files to exclude. e.g. ['m48-2019-11-18-112102.txt', 'm48_DMS_R-2019-11-18-112050.ppd',
               'm53-2019-11-29-093422.txt', 'm53_DMS_R-2019-11-29-093413.ppd']
    save_folder: if [], variables won't be saved, if path of a folder, it will save the variables there
    save_names: names the variables will be saved
    return_variables: if True, it will return a dictionary with the data:
      {sessions: list with all the sessions,
      photometry: list with all the photometry data for each session,
      sample_times_pho: list of the times for each photometry timepoint in the 'photometry clock' (pyPhotometry),
      sample_times_pyc: list of the times for each photometry timepoint in the 'behavioural clock' (pyControl),
      corrected_photometry: list of the (bleached and motion) corrected photometry signals
      photometry_simplified_info: a simplified version of the photometry variable
  '''
  exclusion = kwargs.pop('exclusion', [])
  save_folder = kwargs.pop('save_folder', [])
  save_names = kwargs.pop('save_folder', ['sessions', 'all_photo_data', 'all_sample_times_pho', 'all_sample_times_pyc',
                                         'all_corrected_signal', 'all_photo_data_info'])
  return_variables = kwargs.pop('return_variables', False)

  all_sessions_path, all_photometry_path = pi.import_sessions_photometry_path(
    dir_folder_sessions, dir_folder_photometry, start_str='m', sessions_format='m{id}-{datetime}.txt',
    photometry_format='m{id}_{region}_{hemisphere}-{datetime}.ppd', mouse=mice, day=day, region=[], hemisphere=[],
    exclusion=exclusion)

  sessions = [di.Session(s_path) for s_path in all_sessions_path]
  pp.enable_multiprocessing()
  all_photo_data, all_sample_times_pho, all_sample_times_pyc, all_corrected_signal = \
    zip(*pp.starmap(pi.sync_photometry_data, zip(all_photometry_path,
                                                 np.repeat('m{id}_{region}_{hemisphere}', len(all_photometry_path)),
                                                 sessions, np.repeat(5, len(all_photometry_path)),
                                                 np.repeat(0.001, len(all_photometry_path)))))
  pp.disable_multiprocessing()
  all_photo_data_info = pi.session_information_from_photo_data(all_photo_data)

  if save_folder != []:
    sv.save_variable_joblib(sessions, '{}'.format(save_names[0]), save_folder)
    sv.save_variables_separately_joblib([all_photo_data, all_sample_times_pho, all_sample_times_pyc,
                                         all_corrected_signal, all_photo_data_info],
                                        save_names[1:], save_folder)

  if return_variables:
    return {'sessions': sessions,
            'photometry': all_photo_data,
            'sample_times_pho': all_sample_times_pho,
            'sample_times_pyc': all_sample_times_pyc,
            'corrected_photometry': all_corrected_signal,
            'photometry_simplified_info': all_photo_data_info}

# ---------------------------------------------------------------------------------------------------------------
# 2- Time-wrap trials
# ---------------------------------------------------------------------------------------------------------------
def time_wrap_trials(sessions_together, all_sample_times_pyc_together, all_corrected_signal_together,
                     all_photo_data_info, **kwargs):
  '''
  Time-wrap each trial across all the sessions - import and/or save. Note: data from all cohorts and animals need
  to be together to be able to analyse all together later on
  :param sessions_together: list of all the sessions from all animals
  :param all_sample_times_pyc_together: list of the times for each photometry file in the 'behavioural clock'
  :param all_corrected_signal_together: list of the (bleached and motion) corrected photometry signals
  :param all_photo_data_info: list with all the information of each photometry session
  :param kwargs:
    time_start: time to start aligning. Default=-500: takes 500ms before trial initiation
    time_end: time to end aligning. Default=2500: take 2500ms after outcome delivery
    save_folder: folder to save variables, if empty, variables won't be saved
    save_names: names that variables will be saved
    return_variables: if True, the computed variables will be returned as a dictionary"
      {t_scale_whole: list of aligned trial times,
      pho_scale_together: list of aligned photometry signals,
      v_line: timepoints where aligning happened (at behavioural events),
      dict_median_time: median time between aligned events,
      dict_median_len: median timepoints between aligned events,
      time_start: time start of alignment,
      time_end: time end of alignment,
      all_photo_data_info_short: list with photometry information of each photometry signal
      }
  '''
  time_start = kwargs.pop('time_start', -500)
  time_end = kwargs.pop('time_end', 2500)
  save_folder = kwargs.pop('save_folder', [])
  save_names = kwargs.pop('save_names', ['t_scale_whole', 'pho_scale_together', 'v_line', 'dict_median_time',
                                 'dict_median_len', 'time_start', 'time_end', 'all_photo_data_info_short'])
  return_variables = kwargs.pop('return_variables', False)

  t_scale_whole, pho_scale_together, v_line, dict_median_time, dict_median_len = plp.get_scaled_photometry_per_trial(
    sessions_together, all_sample_times_pyc_together, all_corrected_signal_together, time_start, time_end)
  all_photo_data_info_short = pi.session_information_from_photo_data(all_photo_data_info, keys='short')

  if save_folder != []:
    sv.save_variables_separately_joblib(
      [t_scale_whole, pho_scale_together, v_line, dict_median_time, dict_median_len, time_start, time_end,
       all_photo_data_info_short], save_names, save_folder)

  if return_variables:
    return {'t_scale_whole': t_scale_whole,
            'pho_scale_together': pho_scale_together,
            'v_line': v_line,
            'dict_median_time': dict_median_time,
            'dict_median_len': dict_median_len,
            'time_start': time_start,
            'time_end': time_end,
            'all_photo_data_info_short': all_photo_data_info_short}

# ---------------------------------------------------------------------------------------------------------------
# 3- Dictionary structure for analysis
# ---------------------------------------------------------------------------------------------------------------
def create_indictor_region_dict(save_folder, **kwargs):
  '''
  Create a dictionary with selected data for easy manipulation.
    The default attributes selects the data from the manuscript.
    It is organised so data is split by indicator and region recorded
  :param save_folder: folder path where time-wrap trials were saved
  :param kwargs:
    save_names: file names of the saved variables in save_folder
      Note: save_folder, and save_names come from function above (time_wrap_trials) -
      Therefore, requires to have saved the variables from time_wrap_trials
    dict_name: list of key names of the dictionary
    mice: list of lists of the mice selected for each dict_name keys
    regions: list of lists of the selected region for each dict_name keys
    exclude: list of lists of specific recording sessions to exclude because of technical recording issues.
      [['animal_id', 'region', 'hemisphere', 'date']]
    save_dict_name: string with the file name to be saved. e.g. save_dict_name='dict_photo'
    dict_photo = {indictor_region: {mice: list of mice from the selected sessions,
                                    region: list with the regions from the selected sessions,
                                    hemisphere: list with the recorded hemisphere from each selected sessions,
                                    days: list with the day from each selected sessions,
                                    sessions_select: list of the selected sessions,
                                    all_photo_data_select: list with the photometry data info of the selected sessions,
                                    pho_scale_together_select: list arrays of the time-warped trials for each selected session,
                                    idx_select_sessions: index from the full sessions variable of the sessions selected,
                                    z_score_select: list of the z_scored pho_scale_together_select
                                    }
                  }
  '''
  save_names = kwargs.pop('save_names', ['t_scale_whole', 'pho_scale_together', 'v_line', 'dict_median_time',
                                            'dict_median_len', 'time_start', 'time_end', 'all_photo_data_info_short'])
  dict_name = kwargs.pop('dict_name', ['gcamp_vta', 'gcamp_nac', 'gcamp_dms', 'dlight_nac', 'dlight_dms'])
  mice = kwargs.pop('mice', [[20, 21, 24, 25, 26, 27, 28, 51, 52, 54, 55, 56],
                              [20, 21, 24, 25, 26, 27, 28, 51, 54, 55, 56],
                              [20, 21, 24, 25, 26, 27, 28, 51, 52, 54, 55, 56],
                              [48, 49, 53, 61, 62],
                              [48, 49, 53, 57, 61, 62]])
  regions = kwargs.pop('regions', [['VTA'], ['NAc'], ['DMS'], ['NAc'], ['DMS']])
  exclude = kwargs.pop('exclude', [['21', 'DMS', 'R', '2019-04-28'],
                                   ['24', 'NAc', 'R', '2019-04-29'],
                                   ['24', 'VTA', 'R', '2019-05-21'],
                                   ['27', 'VTA', 'L', '2019-05-08'],
                                   ['28', 'VTA', 'R', '2019-05-02'],
                                   ['28', 'DMS', 'L', '2019-05-11'],
                                   ['51', 'NAc', 'L', '2019-11-19'],
                                   ['51', 'DMS', 'R', '2019-11-20'],
                                   ['51', 'NAc', 'L', '2019-11-21'],
                                   ['51', 'DMS', 'R', '2019-11-23'],
                                   ['51', 'NAc', 'L', '2019-11-24'],
                                   ['51', 'DMS', 'R', '2019-11-26'],
                                   ['51', 'NAc', 'L', '2019-11-27'],
                                   ['51', 'DMS', 'R', '2019-11-29'],
                                   ['54', 'NAc', 'R', '2019-11-19'],
                                   ['54', 'DMS', 'L', '2019-11-20'],
                                   ['54', 'NAc', 'R', '2019-11-22'],
                                   ['54', 'DMS', 'L', '2019-11-23'],
                                   ['54', 'NAc', 'R', '2019-11-25'],
                                   ['54', 'DMS', 'L', '2019-11-26'],
                                   ['54', 'DMS', 'L', '2019-11-29'],
                                   ['55', 'DMS', 'L', '2019-12-10'],
                                   ['55', 'DMS', 'L', '2019-12-13'],
                                   ['56', 'VTA', 'R', '2019-12-12'],
                                   ['57', 'DMS', 'L', '2019-11-30'],
                                   ['57', 'DMS', 'L', '2019-12-01'],
                                   ['57', 'DMS', 'L', '2019-12-02'],
                                   ['57', 'DMS', 'L', '2019-12-03'],
                                   ['57', 'DMS', 'L', '2019-12-04'],
                                   ['61', 'NAc', 'R', '2019-11-30'],
                                   ['61', 'DMS', 'L', '2019-12-01'],
                                   ['61', 'NAc', 'R', '2019-12-02'],
                                   ['61', 'DMS', 'L', '2019-12-03'],
                                   ['61', 'NAc', 'R', '2019-12-04'],
                                   ['62', 'DMS', 'L', '2019-11-18'],
                                   ['62', 'NAc', 'R', '2019-11-19'],
                                   ['62', 'DMS', 'L', '2019-11-20'],
                                   ['62', 'DMS', 'L', '2019-11-21'],
                                   ['62', 'NAc', 'R', '2019-11-22'],
                                   ['62', 'DMS', 'L', '2019-11-23'],
                                   ['62', 'DMS', 'L', '2019-11-24'],
                                   ['62', 'DMS', 'L', '2019-11-25'],
                                   ['62', 'DMS', 'L', '2019-11-26'],
                                   ['62', 'DMS', 'L', '2019-11-27'],
                                   ['62', 'DMS', 'L', '2019-11-28'],
                                   ['62', 'DMS', 'L', '2019-11-29']])
  save_dict_name = kwargs.pop('save_dict_name', [])

  #import previously saved variables (from time_wrap_trials function above)
  t_scale_whole, pho_scale_together, v_line, time_start, time_end, all_photo_data_info, all_sample_times_pyc_together, \
  all_corrected_signal_together, sessions_together = sv.import_several_variables_joblib(save_names, save_folder)

  dict_photo = plp.dict_selected_pho_variables(
    dict_name=dict_name, sessions_together=sessions_together,
    all_photo_data_info=all_photo_data_info, pho_scale_together=pho_scale_together, all_corrected_signal_together=all_corrected_signal_together,
    all_sample_times_pyc= all_sample_times_pyc_together,
    mice=mice,
    regions=regions,
    exclude=exclude)

  if save_dict_name != []:
    sv.save_variable(dict_photo, save_dict_name, save_folder)

# ---------------------------------------------------------------------------------------------------------------
# 4- Import saved variables - provided in data_variables folder (no need to run previous sections 1-3)
# ---------------------------------------------------------------------------------------------------------------
def import_manuscript_variables(dir_folder_save_variables):
  t_scale_whole, v_line, time_start, time_end = sv.import_several_variables_joblib(
    ['t_scale_whole', 'v_line', 'time_start', 'time_end'], dir_folder_save_variables)
  sessions_together = sv.import_variable('sessions_together_select', dir_folder_save_variables)
  dict_photo = sv.import_variable('dict_photo', dir_folder_save_variables)

  sessions = dict_photo['gcamp_vta']['sessions_select'] + dict_photo['gcamp_nac']['sessions_select'] \
             + dict_photo['gcamp_dms']['sessions_select'] + dict_photo['dlight_nac']['sessions_select'] \
             + dict_photo['dlight_dms']['sessions_select']

  return t_scale_whole, v_line, time_start, time_end, sessions_together, dict_photo, sessions

# ---------------------------------------------------------------------------------------------------------------
# 5- Figure 1C-F
# ---------------------------------------------------------------------------------------------------------------
def plot_figure1c(session, **kwargs):
  '''
  Example session of choices exponential moving average
  :param session: session to plot. e.g. sessions[40]
  :param kwargs:
    save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  save = kwargs.pop('save', [])

  fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(2.5, 3), sharex=True)
  pl.plot_exp_mov_ave(session, ax=ax0)
  pl.plot_blocks(session, ax=ax1)
  if save != []:
    pl.savefig(save[0], save[1])

def plot_figure1d(sessions_together, **kwargs):
  '''
  Stay probability
  :param sessions_together: list with all the sessions to plot
  :param kwargs:
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
    selection_type: available options= 'start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'
      See import_beh_data.py (select_trials function for description of these three parameters above)
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  select_n = kwargs.pop('select_n', 'all')
  block_type = kwargs.pop('block_type', 'all')
  selection_type = kwargs.pop('selection_type', 'all')
  save = kwargs.pop('save', [])

  stay_probability, stay_probability_mean, stay_probability_sem, num_session, stay_prob_session, subjects = \
    pl.compute_stay_probability(sessions_together, select_n, reward_type=block_type, selection_type=selection_type,
                                return_stay_prob_joint_trials=True, return_subjects=True)

  fig = plt.figure()
  pl.plot_stay_probability(stay_probability_mean, stay_probability_sem, subjects, stay_prob_session,
                           stay_probability_per_session=stay_prob_session, scatter=True, fontsize=8)
  plt.title('\nselection_type:{}, block_type:{}'.format(selection_type, block_type))
  fig.set_size_inches(2, 2.7)
  if save != []:
    pl.savefig(save[0], save[1])

def plot_figure1e(sessions_together, **kwargs):
  '''
  Mixed model regression predicting stay
  :param sessions_together: list with all the sessions to plot
  :param kwargs:
    predictors: list of regression predictors
    formula: string with regression formula
    selection_type: available options= 'start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
      See import_beh_data.py (select_trials function for description of these three parameters above)
    ticks_formula_names: X axis labels
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
    return_results: if True, return dataframe with the output of the mixed model
  '''
  predictors = kwargs.pop('predictors', ['pr_correct', 'pr_choice', 'pr_outcome', 'pr_trans_CR'])
  formula = kwargs.pop('formula', 'cbind(stay, switch) ~ pr_correct + pr_choice + pr_outcome*pr_trans_CR + '
                          '(pr_correct + pr_choice+ pr_trans_CR*pr_outcome | subject_str)')
  selection_type = kwargs.pop('selection_type', 'all')
  select_n = kwargs.pop('select_n', 'all')
  block_type = kwargs.pop('block_type', 'all')
  ticks_formula_names = kwargs.pop('ticks_formula_names',
                                   [['(Intercept)', 'pr_correct1', 'pr_correct2', 'pr_choice1', 'pr_outcome1',
                                     'pr_trans_CR1', 'pr_outcome1:pr_trans_CR1'],
                                    ['Intercept', 'Neutral', 'Correct', 'Bias', 'Outcome',
                                     'Transition', 'Transition x\n Outcome']])
  save = kwargs.pop('save', [])
  return_results = kwargs.pop('return_results', True)

  df_mixed = pl.plot_mixed_model_regression_stay(sessions_together,
                                      predictors=predictors, formula=formula, expand_re=False,
                                      selection_type=selection_type, select_n=select_n, block_type=block_type,
                                      pltfigure=(2, 2.3), title='mixed model regression - Stay', fontsize=8,
                                      regressor_type='str',
                                      ticks_formula_names=ticks_formula_names,
                                      save=[save],
                                      return_results=return_results
                                      )
  if return_results:
    return df_mixed

def plot_figure1f(sessions_together, **kwargs):
  '''
  Lagged logistic regression (using fixed effects only) predicting stay or switch based on outcome and transition
  ocurred from 12 trials back
  :param sessions_together: list with all the sessions to plot
  :param kwargs:
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
    selection_type: available options= 'start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'
      See import_beh_data.py (select_trials function for description of these three parameters above)
      save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  select_n = kwargs.pop('select_n', 'all')
  block_type = kwargs.pop('block_type', 'all')
  selection_type = kwargs.pop('selection_type', 'all')
  save = kwargs.pop('save', [])

  fig = plt.figure()
  pl.plot_log_reg_lagged(sessions_together, selection_type, block_type, select_n,
                         ['rew_com', 'rew_rare', 'non_com', 'non_rare'], lags=12, log=True,
                         sum_predictors=[[1], [2], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], lags_future=False,
                         pltfigure=False, legend_names=['common transition\nrewarded', 'rare transition\nrewarded',
                                                        'common transition\nunrewarded', 'rare transition\nunrewarded'],
                         colors=['orange', 'blue', 'gold', 'deepskyblue'], scatter=False, legend=True)
  plt.title('\nselection_type:{}, block_type:{}'.format(selection_type, block_type))
  fig.set_size_inches(2, 2.3)
  if save != []:
    pl.savefig(save[0], save[1])

# ---------------------------------------------------------------------------------------------------------------
# 6- Figure 1G-J - run directly plotting functions (plot_figure1g_i and plot_figure1h_j) if using variable
# 'dict_sim_agents' provided in data_variables folder, fits are also provided
# ---------------------------------------------------------------------------------------------------------------
def fit_and_save_behavioural_models_fig1(sessions_together):
  '''
  Fit and save behavioural models used in figure 1 on subjects' data. Fits ocurr per subjects separatelly,
  taking all sessions per subject. Priors are not used, and fitting is performed 30 times. Fits and each repeat is saved
  :param sessions_together: list all sessions from all subjects
  '''
  list_agents = [mf.MF(['bs', 'multpersv']),
                 mb.MB(['bs', 'multpersv']),
                 latent_state.Latent_state(['bs', 'multpersv']),
                 mf_forget_diffa.MF_forget_diffa(['bs', 'multpersv']),
                 mb_forget_0_diffa.MB_forget_0_diffa(['bs', 'multpersv']),
                 latent_state_rewasym.Latent_state_rewasym(['bs', 'multpersv'])
                 ]
  mf.agents_fits_per_subject(sessions_together, use_prior=False, multiprocess='pp', repeats=30, save_repeats=True,
                              agents=list_agents)

def simulate_behaviour_from_models(sessions_together, dir_folder_models, **kwargs):
  '''
  Simulate behaviour from subjects' fits, and save or return it as a dictionary containing:
    {model: list of simulated sessions for each animal}
  :param sessions_together: list of all sessions from all animals
  :param dir_folder_models: path of the folder where subjects' fits are saved
  :param kwargs:
    save: save simulated sessions if True
    return_variable: return dictionary with the simulated sessions per behavioural model
  '''
  save = kwargs.pop('save', [])
  return_variable = kwargs.pop('return_variable', True)

  list_agents = [mf.MF(['bs', 'multpersv']),
                 mb.MB(['bs', 'multpersv']),
                 latent_state.Latent_state(['bs', 'multpersv']),
                 mf_forget_diffa.MF_forget_diffa(['bs', 'multpersv']),
                 mb_forget_0_diffa.MB_forget_0_diffa(['bs', 'multpersv']),
                 latent_state_rewasym.Latent_state_rewasym(['bs', 'multpersv'])
                 ]
  models_no_prior_filename = ['fits_' + agent.name + '_no_prior_per_subject.joblib' for agent in list_agents]
  dict_fits_no_prior = pm.import_variables_to_dict_joblib(models_no_prior_filename, folder_path=dir_folder_models)

  dict_sim_agents = {RL_agent.name: [] for RL_agent in list_agents}
  for RL_agent in list_agents:
    population_fits = mf.get_population_fits(dict_fits_no_prior, 'fits_' + RL_agent.name + '_no_prior_per_subject',
                                              sessions_together,
                                              subjects=list(set([x.subject_ID for x in sessions_together])),
                                              compute_means=True)
    mean_trials = [int(np.mean([x.n_trials for x in sessions_together if x.subject_ID == sub])) for sub in
                   population_fits['sID']]
    mean_sessions = [len([x.n_trials for x in sessions_together if x.subject_ID == sub]) for sub in
                     population_fits['sID']]
    dict_sim_agents[RL_agent.name] = sim.sim_sessions_from_pop_fit(RL_agent, population_fits, n_ses=mean_sessions,
                                                                   n_trials=mean_trials,
                                                                   task=sim.Two_step_fixed,
                                                                   include_forced_choice=True,
                                                                   transition_type=population_fits['transition_type'])
    if save:
      sv.save_variable(dict_sim_agents, 'dict_sim_agents', dir_folder_models)
    if return_variable:
      return dict_sim_agents

#import simulated sessions for each agent
dict_sim_agents = sv.import_variable('dict_sim_agents', dir_folder_variables)

def plot_figure1g_i(dict_sim_agents, **kwargs):
  '''
  Stay probability of simulated models
  :param dict_sim_agents: dictionary with simulated sessions per agent
  :param kwargs:
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
    selection_type: available options= 'start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'
      See import_beh_data.py (select_trials function for description of these three parameters above)
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  select_n = kwargs.pop('select_n', 'all')
  block_type = kwargs.pop('block_type', 'all')
  selection_type = kwargs.pop('selection_type', 'all')
  save = kwargs.pop('save', [])

  fig, axs = plt.subplots(2, 3, sharey=True)
  axs = axs.ravel()
  for i, model in enumerate(dict_sim_agents.keys()):
    stay_probability, stay_probability_mean, stay_probability_sem, num_session, stay_prob_session, subjects = \
      pl.compute_stay_probability(dict_sim_agents[model], select_n, reward_type=block_type,
                                  selection_type=selection_type,
                                  return_stay_prob_joint_trials=True, return_subjects=True)

    pl.plot_stay_probability(stay_probability_mean, stay_probability_sem, subjects, stay_prob_session,
                             stay_probability_per_session=stay_prob_session, scatter=True, fontsize=8, stats=False,
                             ax=axs[i])
    axs[i].set_title('{}'.format(model))
  fig.set_size_inches(6, 5.4)

  if save != []:
    pl.savefig(save[0], save[1])

def plot_figure1h_j(dict_sim_agents, **kwargs):
  '''
  Lagged logistic regression (using fixed effects only) predicting stay or switch based on outcome and transition
  ocurred from 12 trials back
  :param dict_sim_agents: dictionary with simulated sessions per agent
  :param kwargs:
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
    selection_type: available options= 'start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'
      See import_beh_data.py (select_trials function for description of these three parameters above)
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  select_n = kwargs.pop('select_n', 'all')
  block_type = kwargs.pop('block_type', 'all')
  selection_type = kwargs.pop('selection_type', 'all')
  save = kwargs.pop('save', [])

  fig, axs = plt.subplots(2, 3, sharey=True)
  axs = axs.ravel()
  for i, model in enumerate(dict_sim_agents.keys()):
    pl.plot_log_reg_lagged(dict_sim_agents[model], selection_type, block_type, select_n,
                           ['rew_com', 'rew_rare', 'non_com', 'non_rare'], lags=12, log=True,
                           sum_predictors=[[1], [2], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], lags_future=False,
                           pltfigure=False, legend_names=['common transition\nrewarded', 'rare transition\nrewarded',
                                                          'common transition\nunrewarded',
                                                          'rare transition\nunrewarded'],
                           colors=['orange', 'blue', 'gold', 'deepskyblue'], scatter=False, legend=True, ax=axs[i])
    axs[i].set_title('{}'.format(model))
  fig.set_size_inches(6, 5.4)

  if save != []:
    pl.savefig(save[0], save[1])

# ---------------------------------------------------------------------------------------------------------------
# 7- Figure 2 & 4F - dopamine z-score plots
# ---------------------------------------------------------------------------------------------------------------
def plot_figure2_traces(t_scale_whole, dict_photo, v_line, time_start, **kwargs):
  '''
  plot photometry traces
  :param t_scale_whole, dict_photo, v_line, time_start: variables from preprocessing steps
    (functions: time_wrap_trials and create_indictor_region_dict)
  :param kwargs:
    indicator_region: key to use from dict_photo. Default: data from VTA GCaMP recordings
    selection_type: available options= 'mov_average_high', 'mov_average_low', 'reward_mov_average_high',
      'reward_mov_average_low', 'reward_mov_average_medium', 'start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
      See import_beh_data.py (select_trials function for description of these three parameters above)
    all_trial_type: list of the trial types to split data. Available options: 'all', 'free_choice', 'forced_choice',
      'left_choice', 'right_choice', 'ipsi_choice', 'contra_choice', 'up_state', 'down_state', 'common_trans', 'rare_trans',
      'reward_trials', 'nonreward_trials', 'reward_trials_l1', 'nonreward_trials_l1', 'reward_trials_l2', 'nonreward_trials_l2',
      'left_rew', 'left_nonrew', 'right_rew', 'right_nonrew', 'up_rew', 'up_nonrew', 'down_rew', 'down_nonrew', 'common_trans_rew',
      'rare_trans_rew','common_trans_nonrew', 'rare_trans_nonrew', 'correct_trials', 'incorrect_trials', 'neutral_trials',
      'good_second_step_trials', 'bad_second_step_trials', 'good_second_step_rew', 'good_second_step_nonrew', 'bad_second_step_rew',
      'bad_second_step_nonrew', 'correct_common_trans', 'correct_rare_trans', 'incorrect_common_trans', 'incorrect_rare_trans',
      'neutral_common_trans', 'neutral_rare_trans', 'correct_common_trans_rew', 'correct_common_trans_nonrew', 'correct_rare_trans_nonrew',
      'correct_rare_trans_rew', 'incorrect_common_trans_nonrew', 'incorrect_common_trans_rew', 'incorrect_rare_trans_rew',
      'incorrect_rare_trans_nonrew', 'neutral_trials_common_trans_rew', 'neutral_trials_common_trans_nonrew', 'neutral_trials_rare_trans_rew',
      'neutral_trials_rare_trans_nonrew', 'correct_rew', 'correct_nonrew', 'incorrect_rew', 'incorrect_nonrew',
      'short_ITI', 'medium_ITI', 'long_ITI', 'direct_choice', 'change_choice', 'ipsi_direct', 'ipsi_change',
      'contra_direct', 'contra_change', 'same_ch_rew1_rew', 'same_ch_nonrew1_rew', 'same_ch_rew1_nonrew',
      'same_ch_nonrew1_nonrew', 'diff_ch_rew1_rew', 'diff_ch_nonrew1_rew', 'diff_ch_rew1_nonrew', 'diff_ch_nonrew1_nonrew',
      'same_ch_to_ssl1_rew1_rew', 'same_ch_to_ssl1_nonrew1_rew', 'same_ch_to_ssl1_rew1_nonrew', 'same_ch_to_ssl1_nonrew1_nonrew',
      'diff_ch_to_ssl1_rew1_rew', 'diff_ch_to_ssl1_nonrew1_rew', 'diff_ch_to_ssl1_rew1_nonrew', 'diff_ch_to_ssl1_nonrew1_nonrew',
      'same_ch_to_ssl1_common_rew1_rew', 'same_ch_to_ssl1_common_nonrew1_rew', 'same_ch_to_ssl1_common_rew1_nonrew',
      'same_ch_to_ssl1_common_nonrew1_nonrew', 'diff_ch_to_ssl1_common_rew1_rew', 'diff_ch_to_ssl1_common_nonrew1_rew',
      'diff_ch_to_ssl1_common_rew1_nonrew', 'diff_ch_to_ssl1_common_nonrew1_nonrew', 'same_ch_to_ssl1_rare_rew1_rew',
      'same_ch_to_ssl1_rare_nonrew1_rew', 'same_ch_to_ssl1_rare_rew1_nonrew', 'same_ch_to_ssl1_rare_nonrew1_nonrew',
      'diff_ch_to_ssl1_rare_rew1_rew', 'diff_ch_to_ssl1_rare_nonrew1_rew', 'diff_ch_to_ssl1_rare_rew1_nonrew',
      'diff_ch_to_ssl1_rare_nonrew1_nonrew', 'same_ss_rew_trials', 'same_ss_nonrew_trials', 'diff_ss_rew_trials',
      'diff_ss_nonrew_trials'
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  indicator_region = kwargs.pop('indicator_region', 'gcamp_vta')
  selection_type = kwargs.pop('selection_type', 'all')
  select_n = kwargs.pop('select_n', 'all')
  block_type = kwargs.pop('block_type', 'all')
  all_trial_type = kwargs.pop('all_trial_type', ['reward_trials', 'nonreward_trials'])
  save = kwargs.pop('save', [])

  plt.figure(figsize=(2.4, 2.7))
  plp.plot_scaled_trials(dict_photo[indicator_region]['sessions_select'], dict_photo[indicator_region]['all_photo_data_select'],
                         t_scale_whole, dict_photo[indicator_region]['z_score_select'], v_line, time_start,
                         all_trial_type, selection_type=selection_type, select_n=select_n, block_type=block_type,
                         plot_legend=True, all_event_lines=True, colors=['C0', 'C5'], ylim=(-1.3, 1.2))
  if save != []:
    pl.savefig(save[0], save[1])

def plot_figure4f_traces(t_scale_whole, dict_photo, v_line, time_start, **kwargs):
  '''
  :param t_scale_whole, dict_photo, v_line, time_start: variables from preprocessing steps
    (functions: time_wrap_trials and create_indictor_region_dict)
  :param kwargs:
    list_indicator_region: list of keys to use from dict_photo
    list_subjects: list of the subjects to use for each key name in list_indicator_region. if False, use all subjects
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
        See import_beh_data.py (select_trials function for description of these three parameters above)
    all_trial_type: list of the trial types to split data. See function above for all available options
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  list_indicator_region = kwargs.pop('list_indicator_region', ['gcamp_vta', 'gcamp_nac', 'gcamp_dms', 'dlight_nac', 'dlight_dms'])
  list_subjects = kwargs.pop('list_subjects', [False, False, [20, 21, 24, 25, 27, 51, 52, 54, 55, 56], False, False])
  select_n = kwargs.pop('select_n', 'all')
  block_type = kwargs.pop('block_type', 'all')
  all_trial_type = kwargs.pop('all_trial_type', ['reward_trials', 'nonreward_trials'])
  save = kwargs.pop('save', [])

  for indicator_region, select_subjects in zip(list_indicator_region, list_subjects):
    if select_subjects == False:
      mice = dict_photo[indicator_region]['mice']
    else:
      mice = select_subjects[:]

    plt.figure(figsize=(2.4, 2.7))
    plp.plot_scaled_trials(dict_photo[indicator_region]['sessions_select'],
                           dict_photo[indicator_region]['all_photo_data_select'],
                           t_scale_whole, dict_photo[indicator_region]['z_score_select'], v_line, time_start,
                           all_trial_type, selection_type='reward_mov_average_low', select_n=select_n,
                           block_type=block_type,
                           plot_legend=False, all_event_lines=True, colors=['indianred', 'indianred'],
                           line_style=['-', '--'],
                           ylim=(-1.3, 1.2), subjects_list=select_subjects)
    plp.plot_scaled_trials(dict_photo[indicator_region]['sessions_select'],
                           dict_photo[indicator_region]['all_photo_data_select'],
                           t_scale_whole, dict_photo[indicator_region]['z_score_select'], v_line, time_start,
                           all_trial_type, selection_type='reward_mov_average_medium', select_n=select_n,
                           block_type=block_type,
                           plot_legend=False, all_event_lines=True, colors=['mediumseagreen', 'mediumseagreen'],
                           line_style=['-', '--'],
                           ylim=(-1.3, 1.2), subjects_list=select_subjects)
    plp.plot_scaled_trials(dict_photo[indicator_region]['sessions_select'],
                           dict_photo[indicator_region]['all_photo_data_select'],
                           t_scale_whole, dict_photo[indicator_region]['z_score_select'], v_line, time_start,
                           all_trial_type, selection_type='reward_mov_average_high', select_n=select_n,
                           block_type=block_type,
                           plot_legend=False, all_event_lines=True, colors=['skyblue', 'skyblue'],
                           line_style=['-', '--'],
                           ylim=(-1.3, 1.2), subjects_list=select_subjects)
    custom_lines = [Line2D([0], [0], color='darkred', lw=4),
                    Line2D([0], [0], color='lightcoral', lw=4, ls=':'),
                    Line2D([0], [0], color='darkgreen', lw=4),
                    Line2D([0], [0], color='yellowgreen', lw=4, ls=':'),
                    Line2D([0], [0], color='royalblue', lw=4),
                    Line2D([0], [0], color='skyblue', lw=4, ls=':'),
                    ]
    plt.legend(custom_lines, ['rewarded-low', 'unrewarded-low', 'rewarded-medium', 'unrewarded-medium', 'rewarded-high',
                              'unrewarded-high'])
    plt.title('{} - {}'.format(indicator_region, mice))

    if save:
      pl.savefig(save[0], '{}_{}'.format(save[1], indicator_region))


# ---------------------------------------------------------------------------------------------------------------
# 8- Figure 3, 4, S4-8 - Photometry regression
# ---------------------------------------------------------------------------------------------------------------
def plot_photometry_regression(dict_photo, t_scale_whole, v_line, time_start, **kwargs):
  '''
  :param dict_photo, t_scale_whole, v_line, time_start: variables from preprocessing steps
    (functions: time_wrap_trials and create_indictor_region_dict)
  :param kwargs:
    predictors: name of the type of regression to perform. Available options: behavioural, behavioural_by_reward, model
    RL_agent: if 'model' in predictor. RL_agent to use. e.g. latent_state_rewasym.Latent_state_rewasym(['bs', 'multpersv'])
    folder_model: if 'model' in predictor, folder path where the model fits are.
    sessions_together: if 'model' in predictor, sessions object
    regression_type: name of the regression to use. Available options: Linear, Lasso, LassoCV, OLS
    effect_size: plot effect size instead of p values
    block_type: available options= 'all', 'neutral', 'non_neutral', 'non_neutral_after_neutral', 'non_neutral_after_non_neutral'
      See import_beh_data.py (select_trials function for description of these three parameters above)
    selection_type: available options= 'mov_average_high', 'mov_average_low', 'reward_mov_average_high',
      'reward_mov_average_low', 'reward_mov_average_medium', 'start', 'start_1', 'end', 'xtr', 'all', 'xmid', 'xtrrw'
    select_n: if selection_type different from 'all', number of trials to include in the analysis from selection_type
    forced_choice: if True, include all trials, if False, remove forced choice trials
    list_indicator_region: list of the key names from dict_photo to run the photometry
    list_indicator: list if the indicator for each name in list_indicator_region
    list_subjects: list of the subjects to use for each key name in list_indicator_region. if False, use all subjects
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
    return_variables: if True, returns regression coefficients and the alphas used if LassoCV
  '''

  predictors = kwargs.pop('predictors', 'behavioural')
  RL_agent = kwargs.pop('RL_agent', None)
  folder_model = kwargs.pop('folder_model', None)
  sessions_together = kwargs.pop('sessions_together', None)
  regression_type = kwargs.pop('regression_type', 'LassoCV')
  effect_size = kwargs.pop('effect_size', True)
  block_type = kwargs.pop('block_type', 'all')
  selection_type = kwargs.pop('selection_type', 'all')
  select_n = kwargs.pop('select_n', 'all')
  forced_choice = kwargs.pop('forced_choice', True)
  list_indicator_region = kwargs.pop('list_indicator_region', ['gcamp_vta', 'gcamp_nac', 'gcamp_dms', 'dlight_nac', 'dlight_dms'])
  list_indicator = kwargs.pop('list_indicator', ['GCaMP','GCaMP','GCaMP', 'dLight', 'dLight'])
  list_subjects = kwargs.pop('list_subjects', [False, False, [20, 21, 24, 25, 27, 51, 52, 54, 55, 56], False, False])
  save = kwargs.pop('save', [])
  return_variables = kwargs.pop('return_variables', False)

  if predictors == 'behavioural':
    base_predictors = [
      'reward',
      'second_step_update_same', 'second_step_update_diff',
      'model_free_update', 'model_based_update',
      'good_ss',
      'reward_rate',
      'contralateral_choice', 'up_down',
      'correct_choice',
      'repeat_choice',
      'forced_choice_single',
      'common_rare',
      'reward_1',
      'good_ss_1',
    ]

    subplots = [
      ['reward'],
      ['second_step_update_same', 'second_step_update_diff'],
      ['model_free_update', 'model_based_update'],
      ['good_ss', 'good_ss_1', 'correct_choice', 'repeat_choice'],
      ['contralateral_choice', 'up_down'],
      ['forced_choice_single'],
      ['common_rare'],
      ['reward_1', 'reward_rate']
    ]

  elif predictors == 'behavioural_by_reward':
    base_predictors = [
      'reward',
      'second_step_update_rew', 'second_step_update_nonrew',
      'model_free_update', 'model_based_update_rewonly',
      'good_ss',
      'reward_rate',
      'contralateral_choice', 'up_down',
      'correct_choice',
      'repeat_choice',
      'forced_choice_single',
      'common_rare',
      'model_based_update_nonrewonly',
      'reward_1',
      'good_ss_1',
    ]

    subplots = [
      ['reward', 'reward_1', 'reward_rate'],
      ['model_free_update', 'model_based_update_rewonly', 'model_based_update_nonrewonly'],
      ['second_step_update_rew', 'second_step_update_nonrew'],
      ['good_ss', 'correct_choice', 'repeat_choice', 'good_ss_1'],
      ['contralateral_choice', 'up_down'],
      ['forced_choice_single'],
      ['common_rare']
    ]

  elif predictors == 'model':
    model_no_prior_filename = 'fits_' + RL_agent.name + '_no_prior_per_subject.joblib'
    dict_fits = pm.import_variables_to_dict_joblib(model_no_prior_filename, folder_path=folder_model)
    fits_key = 'fits_{}_no_prior_per_subject'.format(RL_agent.name)

    mean_params = np.mean(dict_fits[fits_key]['params'], axis=0)
    Q_val = [RL_agent.session_likelihood(session, mean_params, get_Qval=True) for session in sessions_together]


  all_alpha = []
  coef_reg_all = {indicator_region: [] for indicator_region in list_indicator_region}
  for indicator_region, indicator, select_subjects in zip(list_indicator_region, list_indicator, list_subjects):
    if select_subjects == False:
      mice = dict_photo[indicator_region]['mice']
    else:
      mice = select_subjects[:]
    title = '{} - {} {} - {} block - {}'.format(mice, dict_photo[indicator_region]['region'],
                                                indicator, selection_type, block_type)

    if predictors == 'model':
      Q_values_select = [Q_val[x] for x in dict_photo[indicator_region]['idx_select_sessions']]
      fits_select = [mean_params for x in dict_photo[indicator_region]['idx_select_sessions']]
      fits_names = dict_fits[fits_key]['param_names']
      fits = [fits_names, fits_select]
    else:
      Q_values_select = []
      fits = []

    coef_reg_all[indicator_region], coef_mean, coef_std, coef_sem, coef_t, coef_prob, alpha_ls = \
      plp.plot_photometry_regression(dict_photo[indicator_region]['sessions_select'],
                                     dict_photo[indicator_region]['all_photo_data_select'], t_scale_whole,
                                     dict_photo[indicator_region]['z_score_select'],
                                     v_line, time_start, selection_type, select_n,block_type, base_predictors,
                                     lags={}, single_lag={}, forced_choice=forced_choice, sum_predictors=False,
                                     title=title, text_box=False, contrasts=[], plot_legend=True, return_coef=True,
                                     regression_type=regression_type, subplots=subplots, bootstrap=[], n_permutations=50,
                                     parallel_process=False, ttest=True, multitest=True, test_visual='dots',
                                     subsampled_ttest=20, figsize=(2, 10.3), plot_correlation=False, Q_values=Q_values_select,
                                     extra_predictors=[], fits=fits, all_event_lines=True, effect_size=effect_size,
                                     subjects_list=select_subjects, zscoring_variables=['all'])
    all_alpha.append(alpha_ls)

    if save:
      pl.savefig(save[0],'{}_{}'.format(save[1], indicator_region))

    if return_variables:
      return coef_reg_all, all_alpha

# ---------------------------------------------------------------------------------------------------------------
# 9- Figure 5 - Optogenetics - no need to run import_opto_data function if using variable 'df_stim_opto' provided
# in data_variables folder
# ---------------------------------------------------------------------------------------------------------------
def plot_figure5b(dir_folder, **kwargs):
  '''
  ICSS
  :param dir_folder: data folder path
  :param kwargs:
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']
  '''
  save = kwargs.pop('save', [])
  pl.plot_all_ICSS(dir_folder, figsize=(1.2, 2), scatter=False, boxplot=True)
  if save:
    pl.savefig(save[0], save[1])

def import_opto_data(dir_folder, **kwargs):
  '''
  Import behavioural sessions with optogenetic stimulation
  :param dir_folder: folder where behavioural files are saved
  :param kwargs:
    subjects_virus: dictionary with the virus each animal was injected {subject ID, virus}
    sessions_to_exclude: dictionary with a list with dates to exclude from each subject {subject ID, list dates}
      e.g. {63: ['2020-12-05', '2020-11-29'], 64: ['2020-11-20']}
    save_folder: path of the folder where to save the dataframe with the sessions information, if empty, do not save
    return_dataframe: if True return dataframe with the sessions information
  '''
  subjects_virus = kwargs.pop('subjects_virus', {63: 'YFP',
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
                                                 75: 'YFP'})
  sessions_to_exclude = kwargs.pop('sessions_to_exclude', {})
  save_folder = kwargs.pop('save_folder', [])
  return_dataframe = kwargs.pop('return_dataframe', True)

  df_sessions_opto_activ = pl.df_import_opto_sessions_cohort(dir_folder, subjects_virus,
                                                       sessions_to_exclude=sessions_to_exclude)
  if save_folder:
    sv.save_variable(df_sessions_opto_activ, 'df_sessions_opto_activ', save_folder)

  if return_dataframe:
    return df_sessions_opto_activ

#import dataframe with the optogenetic behavioural data
df_sessions_opto_activ = sv.import_variable('df_stim_opto', dir_folder_variables)

def plot_figure5e_h(df_sessions_opto_activ, **kwargs):
  '''
  Plot latency to initiate next trial
  :param df_sessions_opto_activ: data dataframe
  :param kwargs:
    virus_str_1: string of the virus to plot
    virus_str_2: string of the second virus to plot e.g. 'ChR2'
    stim_type_list: list of the stim types to plot e.g. ['ss_cue', 'outcome_cue']
    ylim_plots: list with the min and max y limits.
    labels: stim types labels.
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']. default: don't save
    dict_subjects_colors: dictionary with the color each animal will be plotted. {animal_id: color}
  '''
  virus_str_1 = kwargs.pop('virus_str_1', 'YFP')
  virus_str_2 = kwargs.pop('virus_str_2', 'ChR2')
  stim_type_list = kwargs.pop('stim_type_list', ['ss_cue', 'outcome_cue'])
  ylim_plots = kwargs.pop('ylim_plots', [100, 660])
  labels = kwargs.pop('labels', ['ss cue', 'outcome cue'])
  save = kwargs.pop('save', [])
  dict_subjects_colors = kwargs.pop('dict_subjects_colors', {63: 'goldenrod',
                                                             64: 'red',
                                                             65: 'violet',
                                                             66: 'yellowgreen',
                                                             67: 'purple',
                                                             69: 'darkorange',
                                                             70: 'blue',
                                                             71: 'gold',
                                                             72: 'dodgerblue',
                                                             73: 'cyan',
                                                             74: 'pink',
                                                             75: 'green'})

  virus1_list = [virus_str_1] * len(stim_type_list)
  virus2_list = [virus_str_2] * len(stim_type_list)

  df_sessions_list = [df_sessions_opto_activ] * len(stim_type_list)
  subjects_colors_list = [dict_subjects_colors] * len(stim_type_list)
  ylim_list = [ylim_plots] * len(stim_type_list)

  for df_sessions, stim_type, virus1, virus2, subjects_colors, ylim in zip(df_sessions_list, stim_type_list,
                                                                           virus1_list,
                                                                           virus2_list, subjects_colors_list,
                                                                           ylim_list):
    pl.plot_beh_correlates_stim_vs_nostim([np.asarray(df_sessions[(df_sessions.stim_type == stim_type) &
                                                                  (df_sessions.virus == virus1)]['sessions']),
                                           np.asarray(df_sessions[(df_sessions.stim_type == stim_type) &
                                                                  (df_sessions.virus == virus2)]['sessions'])],
                                          'latency_initiate',
                                          labels, color_per_subject=subjects_colors,
                                          title='{}'.format(stim_type),
                                          scatter=True, ylabel='latency to initiate (ms)', ylim=ylim,
                                          figsize=(2.5, 2.3))
    if save:
      pl.savefig(save[0], '{}_{}_{}_{}'.format(save[1], virus1, virus2, stim_type))

def figure_5f_i(df_sessions_opto_activ, **kwargs):
  '''
  :param df_sessions_opto_activ: data dataframe
  :param kwargs:
    stim_list: list of the stim types to plot
    virus_list: list of the virus to plot for each entry in stim_list
    save: save if list with folder name, and file name. e.g. ['folder_name', 'figure_name']. default: don't save
    save_folder: folder path to save the regression output, if empty, don't save
    save_name: name file to save the regression output
  '''
  stim_list = kwargs.pop('stim_list', ['ss_cue', 'outcome_cue', 'ss_cue', 'outcome_cue'])
  virus_list = kwargs.pop('virus_list', ['YFP', 'YFP', 'ChR2', 'ChR2'])
  save = kwargs.pop('save', [])
  save_folder = kwargs.pop('save_folder', [])
  save_name = kwargs.pop('save_name', [])

  subjects_virus_activ = {63: 'YFP',
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

  df_sessions_opto_list = [df_sessions_opto_activ] * len(stim_list)
  subjects_virus_list = [subjects_virus_activ] * len(stim_list)

  for df_sessions_opto, subjects_virus, stim, virus in zip(df_sessions_opto_list, subjects_virus_list, stim_list,
                                                           virus_list):
    print('{} {}'.format(stim, virus))
    data = np.asarray(
      df_sessions_opto[(df_sessions_opto.stim_type == stim) & (df_sessions_opto.virus == virus)]['sessions'])

    df_mixed_single = pl.plot_mixed_model_regression_stay(data, predictors=['pr_correct', 'pr_choice', 'pr_outcome',
                                                                            'pr_trans_CR', 'pr_stim'],
                                                          formula='cbind(stay, switch) ~ pr_correct + pr_choice + pr_outcome*pr_trans_CR*pr_stim + '
                                                                  '(pr_correct + pr_choice + pr_outcome*pr_trans_CR*pr_stim || subject_str)',
                                                          expand_re=True, selection_type='all', select_n='all',
                                                          block_type='all',
                                                          pltfigure=(2.2, 2.7), title='mixed model regression - Stay',
                                                          fontsize=8, opto=True,
                                                          regressor_type='str',
                                                          ticks_formula_names=[
                                                            ['(Intercept)', 'pr_correct1', 'pr_correct2', 'pr_choice1',
                                                             'pr_outcome1',
                                                             'pr_trans_CR1', 'pr_stim1',
                                                             'pr_outcome1:pr_trans_CR1', 'pr_outcome1:pr_stim1',
                                                             'pr_trans_CR1:pr_stim1',
                                                             'pr_outcome1:pr_trans_CR1:pr_stim1'],
                                                            ['Intercept', 'Neutral', 'Correct', 'Bias', 'Outcome',
                                                             'Transition',
                                                             'Stimulation',
                                                             'Transition x\n Outcome', 'Outcome x\n Stim',
                                                             'Transition x\n Stim',
                                                             'Transition x\n Outcome x\n Stim']],
                                                          plot_separately=[[0, 1, 2, 3, 4, 5, 7], [6, 8, 9, 10]],
                                                          ylim=[[], []],
                                                          save=save,
                                                          return_results=True, subjects_virus=subjects_virus)
    if save_folder:
      sv.save_variable(df_mixed_single, '{}_{}_{}_full_random_eff'.format(save_name, stim, virus), save_folder)
