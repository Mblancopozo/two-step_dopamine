# -------------------------------------------------------------------------------------
# Code to perform model fitting
# Marta Blanco-Pozo, 2023
# Adapted from Akam, Costa & Dayan (2015), Simple Plans or Sophisticated Habits? State, Transition and Learning Interactions in the Two-Step Task
# -------------------------------------------------------------------------------------
import sys
import numpy as np
from scipy.stats import gamma, beta, norm, laplace
from functools import partial
import scipy.optimize as op
import joblib

from Code_final_manuscript.code import parallel_processing as pp

# -------------------------------------------------------------------------------------
# Maximum likelihood fitting.
# -------------------------------------------------------------------------------------
# Prior distributions
beta_prior = beta(a=2, b=2)  # Prior for unit range parameters.
gamma_prior = gamma(a=2, scale=0.5)  # Prior for positive range parameters.
gamma_prior2 = gamma(a=1, scale=5)  # Prior for positive range parameters.
norm_prior = norm(scale=5)  # Prior for unconstrained range paramters.
laplace_prior = laplace(loc=0, scale=3)  # Prior for weight parameters, also we set the bound so the min value = 0

def _log_prior_prob(params_T, agent):
  priorprobs = np.hstack([
    beta_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'unit')[0]]),
    beta_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'high_half_unit')[0]]),
    beta_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'low_half_unit')[0]]),
    gamma_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'pos')[0]]),
    gamma_prior2.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'pos_big')[0]]),
    norm_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'unc')[0]]),
    beta_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'neg_unit')[0]]),
    beta_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'pos_unit')[0]]),
    laplace_prior.logpdf(params_T[np.where(np.asarray(agent.param_ranges) == 'pos_lap')[0]]) + np.log(2),
  ])
  priorprobs[priorprobs < -1000] = -1000  # Protect against -inf.
  return np.sum(priorprobs)


def _neg_log_likelihood_sessions(params_T, sessions, agent):
  return -np.sum([agent.session_likelihood(session, params_T) for session in sessions])

def _neg_log_posterior_prob_sessions(params_T, sessions, agent):
  priorprob = _log_prior_prob(params_T, agent)
  loglik = np.sum([agent.session_likelihood(session, params_T) for session in sessions])
  return -priorprob - loglik

def _get_init_params(param_ranges, set_param_idx_value={}):
  ''' Get initial parameters by sampling from prior probability distributions.'''
  params_T = []
  for i, rng in enumerate(param_ranges):
    if rng == 'unit':
      params_T.append(beta_prior.rvs())
    elif rng == 'neg_unit':
      params_T.append(beta_prior.rvs())
    elif rng == 'pos_unit':
      params_T.append(beta_prior.rvs())
    elif rng == 'high_half_unit':
      params_T.append(beta_prior.rvs())
    elif rng == 'low_half_unit':
      params_T.append(beta_prior.rvs())
    elif rng == 'pos_big':
      params_T.append(gamma_prior2.rvs())
    elif rng == 'pos':
      params_T.append(gamma_prior.rvs())
    elif rng == 'unc':
      params_T.append(norm_prior.rvs())
    elif rng == 'pos_lap':
      params_T.append(laplace_prior.rvs() + np.log(2))
    elif rng == 'custom':
      params_T.append(set_param_idx_value[i])
  return np.array(params_T)

def fit_subject_sessions(sessions, agent, repeats=30, use_prior=False, print_fits=True, save_repeats=False, set_param_value={}):
  '''ML or MAP fit of session using constrained optimisation.
  IMPORTANT: SET PARAMS VALUE IS NOT SET UP FOR CROSS VALIDATION
  '''
  if use_prior:
    fit_func = partial(_neg_log_posterior_prob_sessions, sessions=sessions, agent=agent)
  else:
    fit_func = partial(_neg_log_likelihood_sessions, sessions=sessions, agent=agent)
  bounds = [{'unc': (None, None), 'unit': (0., 1.), 'pos': (0, None), 'pos_lap': (0, None), 'pos_big': (0, None),
             'neg_unit': (-1, 1), 'pos_unit':(sys.float_info.min, 1.), 'low_half_unit': (0., 0.5),
             'high_half_unit': (0.5, 1.)}[param_range]
            for param_range in agent.param_ranges]
  if set_param_value:
    set_param_idx_value = {}
    for i, id in enumerate([agent.param_names.index(x) for x in list(set_param_value.keys())]):
      bounds[id] = (list(set_param_value.values())[i], np.nextafter(list(set_param_value.values())[i],list(set_param_value.values())[i]+1))
      agent.param_ranges[id] = 'custom'
      set_param_idx_value[id] = list(set_param_value.values())[i]
  else:
    set_param_idx_value = {}
  fits = []
  for r in range(repeats):  # Number of fits to perform with different starting conditions.
    fits.append(op.minimize(fit_func, _get_init_params(agent.param_ranges, set_param_idx_value), method='L-BFGS-B',
                            bounds=bounds, options={'disp': False}))
  fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.
  if save_repeats:
    # save fits repeats of each subject
    with open('{}_'.format(sessions[0].subject_ID) + save_repeats, 'wb') as f:
      joblib.dump(fits, f)
    print('{} saved'.format('{}_'.format(sessions[0].subject_ID) + save_repeats))
  if print_fits:
    print('First sorted fits repeats: {}'.format(np.sort([f['fun'] for f in fits])[:3]))
    print('{}: {}'.format(agent.param_names, fit['x']))
  if use_prior:
    logpostprob = - fit['fun']
    loglik = -_neg_log_likelihood_sessions(fit['x'], sessions, agent)
  else:
    logpostprob = None
    loglik = - fit['fun']
  return {'agent_name': agent.name,
          'param_names': agent.param_names,
          'param_ranges': agent.param_ranges,
          'n_params': agent.n_params,
          'sID': sessions[0].subject_ID,
          'session_filename': [session.file_name for session in sessions],
          'params_T': fit['x'],
          'loglik': loglik,
          'logpostprob': logpostprob,
          'n_trials': np.sum([session.n_trials for session in sessions]),
          'BIC': -2 * loglik + np.log(np.sum([session.n_trials for session in sessions])) * agent.n_params,
          'AIC': -2 * loglik + 2 * agent.n_params}

def fit_per_subject(sessions, agent, use_prior, multiprocess=True, save=False, repeats=15, subjects=[], return_fits=False,
                    save_repeats=False, set_param_value={}):
  '''Perform maximum likelihood fitting on a list of sessions and return
      dictionary with fit information per subject.'''
  if not subjects:
    subjects = list(set([x.subject_ID for x in sessions]))
  sessions_sub = [[x for x in sessions if x.subject_ID == sub] for sub in subjects]

  if save_repeats:
    save_repeats = save

  if multiprocess:
    pp.enable_multiprocessing()
    fit_list = pp.map(partial(fit_subject_sessions, agent=agent, use_prior=use_prior, repeats=repeats,
                              save_repeats=save_repeats, set_param_value=set_param_value), sessions_sub)
    pp.disable_multiprocessing()
  else:
    fit_list = [fit_subject_sessions(sessions, agent, use_prior=use_prior, repeats=repeats, save_repeats=save_repeats,
                                     set_param_value=set_param_value) for sessions in sessions_sub]
  del sessions_sub
  fits = {'agent_name': agent.name,
          'param_names': agent.param_names,
          'param_ranges': agent.param_ranges,
          'n_params': agent.n_params,
          'params': np.array([f['params_T'] for f in fit_list]),
          'sID': np.array([f['sID'] for f in fit_list]),
          'session_filename': np.array([f['session_filename'] for f in fit_list]),
          'loglik': np.array([f['loglik'] for f in fit_list]),
          'logpostprob': np.array([f['logpostprob'] for f in fit_list]),
          'n_trials': np.array([f['n_trials'] for f in fit_list]),
          'BIC': np.array([f['BIC'] for f in fit_list]),
          'AIC': np.array([f['AIC'] for f in fit_list])
          }
  del fit_list
  print('start saving')
  if save:
    with open(save, 'wb') as f:
      joblib.dump(fits, f)
  print('{} fitting done'.format(agent.name))
  if return_fits:
    return fits


def agents_fits_per_subject(sessions, agents, use_prior, multiprocess, return_fits=False, repeats=15,
                            save_repeats=False,
                            set_param_value={}, save_name=None):
  '''
  Perform model-fitting per subject for each agent in agents
  '''
  if return_fits == True:
    return [fit_per_subject(sessions, agent, use_prior=use_prior, multiprocess=multiprocess,
                            save=('fits_' + agent.name + '_per_subject.joblib'),
                            return_fits=True, repeats=repeats, save_repeats=save_repeats,
                            set_param_value=set_param_value)
            for agent in agents]
  else:
    if save_name == None:
      name_end = '_per_subject.joblib' if use_prior is True else '_no_prior_per_subject.joblib'
    else:
      name_end = '_per_subject_{}.joblib'.format(
        save_name) if use_prior is True else '_no_prior_per_subject_{}.joblib'.format(save_name)

    [fit_per_subject(sessions, agent, use_prior=use_prior, multiprocess=multiprocess,
                     save=('fits_' + agent.name + name_end), repeats=repeats, save_repeats=save_repeats,
                     set_param_value=set_param_value)
     for agent in agents]

def get_population_fits(dict_fits, fit_name, sessions, subjects='all', compute_means=True):
  '''
  Extract fits or mean fits per subject
  '''
  population_fits = {}
  population_fits['param_names'] = dict_fits[fit_name]['param_names']
  if compute_means is True:
    if subjects == 'all':
      population_fits['means'] = np.asarray([np.mean(dict_fits[fit_name]['params'][dict_fits[fit_name]['sID'] == sub], axis=0)
        for sub in set(dict_fits[fit_name]['sID'])])
      population_fits['var'] = np.asarray([np.var(dict_fits[fit_name]['params'][dict_fits[fit_name]['sID'] == sub], axis=0)
        for sub in set(dict_fits[fit_name]['sID'])])
      population_fits['sID'] = [sub for sub in set(dict_fits[fit_name]['sID'])]
      population_fits['transition_type'] = [sessions[idx].blocks['transition_states'][0] for idx
                                              in [[x.subject_ID for x in sessions].index(id) for id
                                                  in set(dict_fits[fit_name]['sID'])]]
    else:
      population_fits['means'] = np.asarray(
        [np.mean(dict_fits[fit_name]['params'][dict_fits[fit_name]['sID'] == sub], axis=0)
         for sub in subjects])
      population_fits['var'] = np.asarray(
        [np.var(dict_fits[fit_name]['params'][dict_fits[fit_name]['sID'] == sub], axis=0)
         for sub in subjects])
      population_fits['sID'] = subjects
      population_fits['transition_type'] = [sessions[idx].blocks['transition_states'][0] for idx
                                            in [[x.subject_ID for x in sessions].index(id) for id
                                                in subjects]]
  else:
    population_fits['params'] = dict_fits[fit_name]['params']
    population_fits['sID'] = dict_fits[fit_name]['sID']
  return population_fits