# -------------------------------------------------------------------------------------
# Code with functions to plot results from model fittting
# Marta Blanco-Pozo, 2023
# -------------------------------------------------------------------------------------

import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api
import numpy as np
from matplotlib.cbook import boxplot_stats
from statannotations.Annotator import Annotator

from Code_final_manuscript.code import plot_behaviour as pl, model_fitting as mf

def import_variables_to_dict_joblib(variables, folder_path=[]):
  '''
  Return a dictionary with imported variables
  variables: list of file names
  folder_path: path of the folder where variables are saved
  '''
  dict = {}
  for i in range(len(variables)):
    if folder_path:
      variable_save = os.path.join(folder_path, variables[i])
    else:
      variable_save = variables[i]
    with open(variable_save, 'rb') as f:
      dict[variables[i].split('.')[0]] = joblib.load(f)
  return dict

def plot_fitted_param(dict_models, name_fits, title=[], subject_fits_path=None, figsize=(3, 2.5)):
  '''
  Boxplot of the parameter fits
  :param dict_models: dictionary with the fits of each model (each model as a key)
  :param name_fits: name of the model (key in dict_models)
  :param title: title of the figure
  :param subject_fits_path: path to the files containing the best fits for each subject separately,
  will plot then the std of the best param fits
  '''
  df_params = pd.DataFrame(dict_models[name_fits]['params'], columns=dict_models[name_fits]['param_names'],
                           index=[str(x) for x in dict_models[name_fits]['sID']])

  ranges = dict_models[name_fits]['param_ranges']
  param_names = dict_models[name_fits]['param_names']
  unit_names = [param_names[i] for i in range(len(param_names)) if ranges[i] == 'unit' or param_names[i] == 'bs']

  df_params_melt = pd.melt(df_params.reset_index(), id_vars='index', var_name='params', value_name='fitted value')
  df_params_melt.rename(columns={'index': 'subject'}, inplace=True)
  df_params_melt['subject'] = df_params_melt['subject'].astype('category')
  df_params_melt['range'] = ['unit' if x in unit_names else 'non_unit' for x in df_params_melt['params']]


  fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [3, 1]}, figsize=figsize)
  sns.boxplot(x="params", y="fitted value", data=df_params_melt[df_params_melt.range == 'unit'], ax=axs[0],
              color='grey')
  sns.stripplot(x="params", y="fitted value", data=df_params_melt[df_params_melt.range == 'unit'], hue='subject',
                ax=axs[0], palette='tab20')
  axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, horizontalalignment='right')
  axs[0].axhline(0, ls='--')
  sns.boxplot(x="params", y="fitted value", data=df_params_melt[df_params_melt.range == 'non_unit'], ax=axs[1],
              color='grey')
  sns.stripplot(x="params", y="fitted value", data=df_params_melt[df_params_melt.range == 'non_unit'], hue='subject',
                ax=axs[1], palette='tab20')
  axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, horizontalalignment='right')
  axs[1].axhline(0, ls='--')
  axs[0].get_legend().remove()
  axs[1].get_legend().remove()
  handles, labels = axs[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc=7)

  t, p_val = zip(*[stats.ttest_1samp(df_params[name].values, 0)
                   for name in df_params.columns])
  p_val_corrected = statsmodels.stats.multitest.multipletests(p_val, method='bonferroni')
  anotation = pl.stats_annotation(p_val_corrected[1])
  annotation_0 = [anotation[id] for id in np.concatenate(
    [np.where(df_params.columns == x)[0] for x in [x.get_text() for x in axs[0].get_xticklabels()]])]
  annotation_1 = [anotation[id] for id in np.concatenate(
    [np.where(df_params.columns == x)[0] for x in [x.get_text() for x in axs[1].get_xticklabels()]])]

  for i, xtick in enumerate(axs[0].get_xticks()):
    text = annotation_0[i]
    axs[0].text(xtick, axs[0].get_ylim()[-1] + (axs[0].get_yticks()[-1] - axs[0].get_yticks()[-2]) / 4, text,
                horizontalalignment='center', size='x-large', weight='semibold')
  for i, xtick in enumerate(axs[1].get_xticks()):
    text = annotation_1[i]
    axs[1].text(xtick, axs[1].get_ylim()[-1] + (axs[1].get_yticks()[-1] - axs[1].get_yticks()[-2]) / 4, text,
                horizontalalignment='center', size='x-large', weight='semibold')

  if not title:
    title = name_fits
  fig.suptitle(title)
  fig.tight_layout()
  fig.subplots_adjust(right=0.85)

  if subject_fits_path:

    # Import each animal's fits
    names = ['{}_{}.joblib'.format(sub, name_fits) for sub in dict_models[name_fits]['sID']]
    dict_fits_subjects = mf.import_variables_to_dict_joblib(names, folder_path=subject_fits_path)

    names = ['{}_{}'.format(sub, name_fits) for sub in dict_models[name_fits]['sID']]
    subjects = dict_models[name_fits]['sID']
    fun_best = np.concatenate([[dict_fits_subjects[name][idx]['fun'] for idx in
                                np.argsort([f['fun'] for f in dict_fits_subjects[name]])[:3]] for name in names])
    params_best = np.concatenate([[dict_fits_subjects[name][idx]['x'] for idx in
                                   np.argsort([f['fun'] for f in dict_fits_subjects[name]])[:3]] for name in names])

    df_fits_sub_best = pd.DataFrame(np.column_stack((fun_best, params_best)),
                                    index=pd.MultiIndex.from_tuples(
                                      list(zip(*[np.repeat(subjects, 3), np.tile([1, 2, 3], len(subjects))])),
                                      names=['subject', 'repeats']),
                                    columns=['fun'] + dict_models[name_fits]['param_names'])

    df_fits_sub_best.groupby(level='subject')['fun'].std()

    np.asarray([df_fits_sub_best.groupby(level='subject')[col].std().values for col in df_fits_sub_best.columns])

    df_std = pd.DataFrame(
      np.asarray([df_fits_sub_best.groupby(level='subject')[col].std() for col in df_fits_sub_best.columns]).T,
      index=df_fits_sub_best.groupby(level='subject')['fun'].std().index,
      columns=df_fits_sub_best.columns)

    df_std_melt = pd.melt(df_std, var_name='params', value_name='std')

    # Plot sd of the 3 best fits
    fig, ax = plt.subplots()
    ax = sns.boxplot(x="params", y="std", data=df_std_melt)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title(
      'Standard deviation best 3 repeats \n Model: flexible forget vs corr asym + bs + multpersv - {}'.format(title))
    fig.tight_layout()


def plot_ordered_model(dict_models, param_to_plot='loglik', keys_names=[], dict_colors={}, dict_hatches={}, pairs=[],
                       return_df=False, figsize=(4,6), xticks=[], cross_val=False):
  '''
  plot ordered models
  if cross_val=True, dict_models=dataframe of cross-validated data --> this only work for loglik
  '''
  if param_to_plot == 'loglik':
    if cross_val:
      param = np.asarray([dict_models[x].values for x in dict_models.keys()])
      value_name = 'Cross-validated Δ log likelihood'
    else:
      param = np.asarray([dict_models[x]['loglik'] for x in dict_models.keys()])
      value_name = 'Δ log likelihood'
  elif param_to_plot == 'BIC':
    param = np.asarray([dict_models[x]['BIC'] for x in dict_models.keys()])
    value_name = 'Δ BIC'
  elif param_to_plot == 'AIC':
    param = np.asarray([dict_models[x]['AIC'] for x in dict_models.keys()])
    value_name = 'Δ AIC'

  if cross_val:
    subjects = dict_models[list(dict_models.keys())[0]].index
  else:
    subjects = dict_models[list(dict_models.keys())[0]]['sID']

  if not keys_names:
    keys_names = dict_models.keys()

  df_param = pd.DataFrame(param.T, columns=keys_names, index=subjects)

  if param_to_plot == 'loglik':
    df_param_sub = df_param.sub(df_param.max(axis=1), axis=0)
  else:
    df_param_sub = df_param.sub(df_param.min(axis=1), axis=0)
  df_melt = pd.melt(df_param_sub, var_name='agents', value_name=value_name)

  if param_to_plot == 'loglik':
    order_models = df_param_sub.mean().sort_values(ascending=False).index[:]
  else:
    order_models = df_param_sub.mean().sort_values(ascending=True).index[:]

  if dict_colors:
    ordered_agents_colors = [[dict_colors[key] for key in list(dict_colors.keys()) if key in name][0]
                             for name in list(order_models)]
  else:
    ordered_agents_colors = ['grey' for name in list(order_models)]

  plot_params = {
    'data': df_melt,
    'x': 'agents',
    'y': value_name,
    'order': order_models,
    'palette': ordered_agents_colors
  }

  fig, ax = plt.subplots(figsize=figsize)
  box_plot = sns.boxplot(ax=ax, **plot_params)

  t, p_val = zip(*[stats.wilcoxon(df_param_sub[name].values, df_param_sub[order_models[0]].values, zero_method='zsplit')
                   for name in order_models])
  p_val_corrected = statsmodels.stats.multitest.multipletests(p_val, method='bonferroni')
  anotation = pl.stats_annotation(p_val_corrected[1])
  print('Wilcoxon test + bonferroni correction')

  if param_to_plot == 'loglik':
    hi_whisker = boxplot_stats(df_melt[value_name])[0]['whishi']
    add = 0.001
  else:
    hi_whisker = boxplot_stats(df_melt[value_name])[0]['whislo']
    add = -100

  for i, xtick in enumerate(box_plot.get_xticks()):
    text = anotation[i]
    box_plot.text(xtick, hi_whisker + add, text,
                  horizontalalignment='center', size='x-large', weight='semibold')

  if dict_hatches:
    ordered_agents_hatches = [[dict_hatches[key] for key in list(dict_hatches.keys()) if key in name][0]
                              for name in list(order_models)]
    for hatch, patch in zip(ordered_agents_hatches, box_plot.artists):
      patch.set_hatch(hatch)

  if not xticks:
    ax.set_xticklabels(box_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
  else:
    ax.set_xticklabels(xticks)

  if pairs:
    annotator = Annotator(ax, pairs, **plot_params)
    annotator.configure(test="t-test_paired", comparisons_correction="bonferroni", text_format='star', fontsize=12)
    _, corrected_results = annotator.apply_and_annotate()
  fig.tight_layout()

  if return_df:
    return df_param

def plot_parameter_influence(sessions_together, RL_agent, dict_fits_agent, save=False, return_df=False, ratio_sd=False,
                             param_plot=['Q_net', 'Q_hyb', 'Q_c', 'Q_mb_uncorr', 'Q_mb_corr', 'V_s_uncorr', 'V_s_corr'],
                             pairs=[
                               ('Q_net', 'Q_mb_corr'),
                               ('Q_hyb', 'Q_mb_corr'),
                               ('Q_c', 'Q_mb_uncorr'),
                               ('Q_c', 'Q_mb_corr'),
                               ('Q_mb_uncorr', 'Q_mb_corr'),
                               ('V_s_uncorr', 'V_s_corr'),
                             ], figsize=(4, 4)):
  '''
  Plot the influence of each paramter in the model on choice behaviour
  '''
  all_subjects = list(set([x.subject_ID for x in sessions_together]))

  fits_sessions_per_subject = [dict_fits_agent['params'][list(dict_fits_agent['sID']).index(sub)]
                               for sub in [x.subject_ID for x in sessions_together]]
  Q_val = [[RL_agent.session_likelihood(session, fits, get_Qval=True)
            for session, fits in zip(sessions_together, fits_sessions_per_subject) if session.subject_ID == sub]
           for sub in all_subjects]

  params_sub = [dict_fits_agent['params'][list(dict_fits_agent['sID']).index(sub)] for sub in all_subjects]
  param_names = dict_fits_agent['param_names']

  dict = {}
  dict['Q_net'] = [np.mean([np.std(Q_val[sub_id][ses_id]['Q_net'].T[1] - Q_val[sub_id][ses_id]['Q_net'].T[0])
                    for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]

  if 'Q_hyb' in param_plot:
    dict['Q_hyb'] = [np.mean([np.std(Q_val[sub_id][ses_id]['Q_hyb'].T[1] - Q_val[sub_id][ses_id]['Q_hyb'].T[0])
                      for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]

  if 'Q_c' in param_plot:
    if 'weight_mf' in param_names:
      dict['Q_c'] = [np.mean([np.std((Q_val[sub_id][ses_id]['Q_c'].T[1] - Q_val[sub_id][ses_id]['Q_c'].T[0]) *
                           params_sub[sub_id][param_names.index('weight_mf')])
                    for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]
    else:
      dict['Q_c'] = [np.mean([np.std((Q_val[sub_id][ses_id]['Q_c'].T[1] - Q_val[sub_id][ses_id]['Q_c'].T[0]) *
                                     (1 - params_sub[sub_id][param_names.index('weight_mb_uncorr')] -
                                      params_sub[sub_id][param_names.index('weight_mb_corr')]))
                              for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]

  if 'Q_mb_uncorr' in param_plot:
    dict['Q_mb_uncorr'] = [np.mean([np.std((Q_val[sub_id][ses_id]['Q_mb_uncorr'].T[1] - Q_val[sub_id][ses_id]['Q_mb_uncorr'].T[0]) *
      params_sub[sub_id][param_names.index('weight_mb_uncorr')])
                            for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]
  if 'Q_mb_bayes' in param_plot:
    dict['Q_mb_bayes'] = [np.mean([np.std((Q_val[sub_id][ses_id]['Q_mb_bayes'].T[1] - Q_val[sub_id][ses_id]['Q_mb_bayes'].T[0]) *
      params_sub[sub_id][param_names.index('weight_mb_bayes')])
                            for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]
  if 'Q_mb_corr' in param_plot:
    dict['Q_mb_corr'] = [np.mean([np.std((Q_val[sub_id][ses_id]['Q_mb_corr'].T[1] - Q_val[sub_id][ses_id]['Q_mb_corr'].T[0]) *
                                 params_sub[sub_id][param_names.index('weight_mb_corr')])
                          for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]
  if 'V_s_uncorr' in param_plot:
    dict['V_s_uncorr'] = [np.mean([np.std(Q_val[sub_id][ses_id]['V_s_uncorr'].T[1] - Q_val[sub_id][ses_id]['V_s_uncorr'].T[0])
                           for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]
  if 'V_s_corr' in param_plot:
    dict['V_s_corr'] = [np.mean([np.std(Q_val[sub_id][ses_id]['V_s_corr'].T[1] - Q_val[sub_id][ses_id]['V_s_corr'].T[0])
                         for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]
  if 'V_s_bayes' in param_plot:
    dict['V_s_bayes'] = [np.mean([np.std(Q_val[sub_id][ses_id]['V_s_bayes'].T[1] - Q_val[sub_id][ses_id]['V_s_bayes'].T[0])
                         for ses_id in range(len(Q_val[sub_id]))]) for sub_id in range(len(Q_val))]



  if ratio_sd:
    df_std_qval = pd.DataFrame(np.asarray([np.asarray(dict[x])/np.asarray(dict['Q_net']) for x in param_plot]).T,
                               columns=param_plot,
                               index=all_subjects)
  else:

    df_std_qval = pd.DataFrame(np.asarray([np.asarray(dict[x]) for x in param_plot]).T,
                               columns=param_plot,
                               index=all_subjects)

  df_std_melt = pd.melt(df_std_qval.reset_index(), id_vars='index', var_name='Values', value_name='std')
  df_std_melt.rename(columns={'index': 'subject'}, inplace=True)
  df_std_melt['subject'] = df_std_melt['subject'].astype('category')


  plot_params = {
    'data': df_std_melt,
    'x': 'Values',
    'y': 'std',
    "color": "grey",
  }

  fig, ax = plt.subplots(figsize=figsize)
  sns.boxplot(ax=ax, **plot_params)
  sns.stripplot(x="Values", y="std", data=df_std_melt, hue='subject', palette='tab20')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
  if ratio_sd:
    ax.set_ylabel('std/Q net std')
  ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
  ax.set_title('Influence on choices')
  annotator = Annotator(ax, pairs, **plot_params)
  annotator.configure(test="t-test_paired", comparisons_correction="bonferroni", text_format='star', fontsize=12)
  _, corrected_results = annotator.apply_and_annotate()

  if save:
    if ratio_sd:
      pl.savefig(save, 'ration_std_Qval_{}'.format(RL_agent.name))
    else:
      pl.savefig(save, 'std_Qval_{}'.format(RL_agent.name))

  if return_df:
    return df_std_qval

def plot_std_models_session(all_dicts, fits_agent, RL_agent, sessions_together, ses_id=0, sub=24, figsize=(6, 4), ax=None):
  '''
  Plot the standard deviation of RL_agent Q_values of a specific session
  '''
  if not ax:
    fig, ax = plt.subplots(1,1, figsize=figsize)
  fits_sessions_per_subject = [all_dicts[fits_agent]['params'][list(all_dicts[fits_agent]['sID']).index(sub)]
                               for sub in [x.subject_ID for x in sessions_together]]
  all_subjects = list(set([x.subject_ID for x in sessions_together]))
  Q_val, sessions_sub = zip(*[zip(*[(RL_agent.session_likelihood(session, fits, get_Qval=True), session)
            for session, fits in zip(sessions_together, fits_sessions_per_subject) if session.subject_ID == sub])
           for sub in all_subjects])
  param_names = all_dicts[fits_agent]['param_names']

  sub_id = all_subjects.index(sub)

  params_sub = all_dicts[fits_agent]['params'][list(all_dicts[fits_agent]['sID']).index(sub)]
  params_name = all_dicts[fits_agent]['param_names']
  pl.plot_exp_mov_ave(sessions_sub[sub_id][ses_id], ax=ax, color_choices='k', tau=1)
  ax.set_title('')
  ax2 = ax.twinx()
  ax2.plot(Q_val[sub_id][ses_id]['Q_net'].T[1] - Q_val[sub_id][ses_id]['Q_net'].T[0], label='Q net')
  Q_mf = Q_val[sub_id][ses_id]['Q_c'] * params_sub[params_name.index('weight_mf')]
  ax2.plot(Q_mf.T[1] - Q_mf.T[0], label='Q mf')
  Q_mb_uncorr = Q_val[sub_id][ses_id]['Q_mb_uncorr'] * params_sub[params_name.index('weight_mb_uncorr')]
  ax2.plot(Q_mb_uncorr.T[1] - Q_mb_uncorr.T[0], label='Q mb uncorr')
  Q_mb_corr = Q_val[sub_id][ses_id]['Q_mb_corr'] * params_sub[params_name.index('weight_mb_corr')]
  ax2.plot(Q_mb_corr.T[1] - Q_mb_corr.T[0], label='Q mb corr')
  ax2.axhline(0, ls='--', color='k')
  ax2.set_ylabel('∆ Value left - right')
  ax2.set_xlabel('Trial #')
  ax2.set_title('m{} - uncorr {} - corr {}'.format(sub, np.std(Q_mb_uncorr.T[1] - Q_mb_uncorr.T[0]),
                                               np.std(Q_mb_corr.T[1] - Q_mb_corr.T[0])))
  ax2.legend()

