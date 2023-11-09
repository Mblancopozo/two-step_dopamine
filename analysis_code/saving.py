# -------------------------------------------------------------------------------------
# Code with saving functions
# -------------------------------------------------------------------------------------

import os
import pickle
import joblib

def save_variable(variable, name, folder_path):
  with open(os.path.join(folder_path, '{}.pkl'.format(name)), 'wb') as f:
    pickle.dump(variable, f)

def save_variable_joblib(variable, name, folder_path):
  with open(os.path.join(folder_path, '{}.joblib'.format(name)), 'wb') as f:
    joblib.dump(variable, f)

def save_variables_separately(variables, names, folder_path):
  for variable, name in zip(variables, names):
    save_variable(variable, name, folder_path)
    print('{} saved'.format(name))

def save_variables_separately_joblib(variables, names, folder_path):
  for variable, name in zip(variables, names):
    save_variable_joblib(variable, name, folder_path)
    print('{} saved'.format(name))

def import_variable(name, folder_path):
  with open(os.path.join(folder_path, '{}.pkl'.format(name)), 'rb') as f:
    return pickle.load(f)

def import_variable_joblib(name, folder_path):
  with open(os.path.join(folder_path, '{}.joblib'.format(name)), 'rb') as f:
    return joblib.load(f)

def import_several_variables(names, folder_path):
  return [import_variable(name, folder_path) for name in names]

def import_several_variables_joblib(names, folder_path):
  return [import_variable_joblib(name, folder_path) for name in names]