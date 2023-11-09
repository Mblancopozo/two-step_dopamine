#----------------------------------------------------------------------------------
# Code to import and pre-process (motion and bleaching correction) photometry data,
# and align photometry and behavioural sessions
# Marta Blanco-Pozo, 2023
#----------------------------------------------------------------------------------

import os
import numpy as np
from parse import *
import glob
import operator
from scipy.signal import medfilt, butter, filtfilt, detrend
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress
from scipy.optimize import curve_fit
import rsync as rs


def als(y, lam=1e4, p=0.05, niter=10):
  L = len(y)
  diag = np.ones(L - 2)
  D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2).tocsc()
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
    print(w)
  return z

def linear_func(x, a, b):
  return a + b * x

def double_exp(x, a, b, c, d):
  return a * np.exp(b * x) + c * np.exp(d * x)

#%% align behaviour and photometry sessions
def import_sessions_photometry_path(dir_folder_session, dir_folder_pho, start_str, sessions_format, photometry_format,
                                    mouse, day, region=[],
                                    hemisphere=[], exclusion=[]):

  '''
  photometry_format: e.g. ''m52_DMS_R-2019-12-18-090859.ppd' --> 'm{id}_{region}_{hemisphere}-{datetime}.ppd'
  '''

  all_sessions_path = glob.glob(os.path.join(dir_folder_session, '{}*'.format(start_str)))
  all_photometry_path = glob.glob(os.path.join(dir_folder_pho, '{}*'.format(start_str)))

  if exclusion != []:
    exclusion_sessions = [os.path.join(dir_folder_session, x) for x in exclusion]
    all_sessions_path = [x for x in all_sessions_path if x not in exclusion_sessions]
    exclusion_photometry = [os.path.join(dir_folder_pho, x) for x in exclusion]
    all_photometry_path = [x for x in all_photometry_path if x not in exclusion_photometry]

  all_sessions_path = [os.path.basename(all_sessions_path[i]) for i in range(len(all_sessions_path))]
  all_photometry_path = [os.path.basename(all_photometry_path[i]) for i in range(len(all_photometry_path))]

  list_sessions_m_d = [
    [parse(sessions_format, all_sessions_path[i])['id'],
     datetime.strptime(parse(sessions_format, all_sessions_path[i])['datetime'], '%Y-%m-%d-%H%M%S')] for i in
    range(len(all_sessions_path))]

  sort_sessions_m_d = sorted(list_sessions_m_d, key=operator.itemgetter(0, 1))
  idx_sort_sessions = [list_sessions_m_d.index(x) for x in sort_sessions_m_d]
  all_sessions_path = [all_sessions_path[x] for x in idx_sort_sessions]

  list_pho_m_d = [
    [parse(photometry_format, all_photometry_path[i])['id'],
     datetime.strptime(parse(photometry_format, all_photometry_path[i])['datetime'], '%Y-%m-%d-%H%M%S')] for i in
    range(len(all_photometry_path))]

  sort_pho_m_d = sorted(list_pho_m_d, key=operator.itemgetter(0, 1))
  idx_sort_pho_m_d = [list_pho_m_d.index(x) for x in sort_pho_m_d]
  all_photometry_path = [all_photometry_path[x] for x in idx_sort_pho_m_d]

  # select animal, day, region, hemisphere to import
  if mouse:
    mouse_id = [int(parse(photometry_format, all_photometry_path[i])['id']) in mouse
                for i in range(len(all_photometry_path))]
    all_photometry_path = [all_photometry_path[i] for i in np.where(mouse_id)[0]]

    mouse_id = [int(parse(sessions_format, all_sessions_path[i])['id']) in mouse
                for i in range(len(all_sessions_path))]
    all_sessions_path = [all_sessions_path[i] for i in np.where(mouse_id)[0]]

  if day:
    day_id = [parse(photometry_format, all_photometry_path[i])['datetime'][:-7] in day
              for i in range(len(all_photometry_path))]
    all_photometry_path = [all_photometry_path[i] for i in np.where(day_id)[0]]
    day_id = [parse(sessions_format, all_sessions_path[i])['datetime'][:-7] in day
              for i in range(len(all_sessions_path))]
    all_sessions_path = [all_sessions_path[i] for i in np.where(day_id)[0]]

  if region:
    region_id = [parse(photometry_format, all_photometry_path[i])['region'] in region
                 for i in range(len(all_photometry_path))]
    all_photometry_path = [all_photometry_path[i] for i in np.where(region_id)[0]]
    all_sessions_path = [all_sessions_path[i] for i in np.where(region_id)[0]]

  if hemisphere:
    hemisphere_id = [parse(photometry_format, all_photometry_path[i])['hemisphere'] in hemisphere
                for i in range(len(all_photometry_path))]
    all_photometry_path = [all_photometry_path[i] for i in np.where(hemisphere_id)[0]]
    all_sessions_path = [all_sessions_path[i] for i in np.where(hemisphere_id)[0]]

  #check behavioural and photometry data are correctly paired
  mouse_check = [parse(sessions_format, all_sessions_path[i])['id'] == parse(photometry_format, all_photometry_path[i])['id']
                 for i in range(len(all_photometry_path))]

  day_check = [parse(sessions_format, all_sessions_path[i])['datetime'][:-7] ==
               parse(photometry_format, all_photometry_path[i])['datetime'][:-7]
               for i in range(len(all_photometry_path))]

  if any(mc is False for mc in mouse_check):
    raise ValueError('Mouse is not correctly aligned')
  elif any(dc is False for dc in day_check):
    raise ValueError('Day is not correctly aligned')
  else:
    all_sessions_path = [os.path.join(dir_folder_session, x) for x in
                         [all_sessions_path[i] for i in range(len(all_sessions_path))]]
    all_photometry_path = [os.path.join(dir_folder_pho, x) for x in
                           [all_photometry_path[i] for i in range(len(all_photometry_path))]]
    return all_sessions_path, all_photometry_path

#%% Photometry preprocessing
def import_data(file_path, subject_format, low_pass=2, high_pass=0.001, order=3, old_format=False,
                signal_name='df'):

  # adapted from https://pyphotometry.readthedocs.io/en/latest/user-guide/importing-data/

  with open(file_path, 'rb') as f:
    header_size = int.from_bytes(f.read(2), 'little')
    data_header = f.read(header_size)
    data = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
  # Extract header information
  if old_format:
    subject_ID = data_header[:12].decode().strip()
    date_time = datetime.strptime(data_header[12:31].decode(), '%Y-%m-%dT%H:%M:%S')
    mode = {1: 'GCaMP/RFP', 2: 'GCaMP/iso', 3: 'GCaMP/RFP_dif'}[data_header[31]]
    sampling_rate = int.from_bytes(data_header[32:34], 'little')
    volts_per_division = np.frombuffer(data_header[34:42], dtype='<u4') * 1e-9
  else:
    try:
      subject_ID = parse(subject_format, data_header.decode().split('",')[0].split()[1][1:])['id']
      if 'hemisphere' in parse(subject_format, data_header.decode().split('",')[0].split()[1][1:]).named.keys():
        hemisphere = parse(subject_format, data_header.decode().split('",')[0].split()[1][1:])['hemisphere']
      else:
        hemisphere = 'NaN'
      if 'region' in parse(subject_format, data_header.decode().split('",')[0].split()[1][1:]).named.keys():
        region = parse(subject_format, data_header.decode().split('",')[0].split()[1][1:])['region']
      else:
        region = 'NaN'
    except AttributeError:
      print('data header: {}'.format(data_header))
      print('Extracting session information from file name')
      subject_ID = parse(subject_format, file_path.split('/')[-1])['id']
      if 'hemisphere' in parse(subject_format, file_path.split('/')[-1]).named.keys():
        hemisphere = parse(subject_format, file_path.split('/')[-1])['hemisphere'][0]
      else:
        hemisphere = 'NaN'
      if 'region' in parse(subject_format, file_path.split('/')[-1]).named.keys():
        region = parse(subject_format, file_path.split('/')[-1])['region']
      else:
        region = 'NaN'

    date_time = datetime.strptime(data_header.decode().split('",')[1].split()[1][1:], '%Y-%m-%dT%H:%M:%S')
    mode = data_header.decode().split('",')[2].split('"')[-1]
    sampling_rate = int(data_header.decode().split('",')[3].split(', "')[0].split()[-1])
    volts_per_division = np.array([float(data_header.decode().split('",')[3].split(', "')[1].split(': ')[-1].
                                         replace(']', '').replace('[', '').split(',')[0]),
                                   float(data_header.decode().split('",')[3].split(', "')[1].split(': ')[-1].
                                         replace(']', '').replace('[', '').split(',')[1])])
  # Extract signals.
  signal = data >> 1  # Analog signal is most significant 15 bits.
  digital = (data % 2) == 1  # Digital signal is least significant bit.
  # Alternating samples are signals 1 and 2.
  ADC1 = signal[::2] * volts_per_division[0]
  ADC2 = signal[1::2] * volts_per_division[1]
  DI1 = digital[::2]
  DI2 = digital[1::2]
  t = np.arange(ADC1.shape[0]) / sampling_rate  # Time relative to start of recording (seconds).
  # Median filtering to remove electrical artifact
  ADC1_denoised = medfilt(ADC1, kernel_size=5)
  ADC2_denoised = medfilt(ADC2, kernel_size=5)
  if signal_name in ['df', 'det']:
    model = np.polyfit(range(len(ADC1_denoised)), ADC1_denoised, order)
    predicted = np.polyval(model, range(len(ADC1_denoised)))
    ADC1_f = ADC1_denoised - predicted
    # detrend ADC2
    model = np.polyfit(range(len(ADC2_denoised)), ADC2_denoised, order)
    predicted = np.polyval(model, range(len(ADC2_denoised)))
    ADC2_f = ADC2_denoised - predicted
  # detrend by substracting a linear least-square fit to the data
  elif signal_name in ['det_sq']:
    ADC1_f = detrend(ADC1_denoised)
    ADC2_f = detrend(ADC2_denoised)
  # Filter signals.
  if low_pass and high_pass:
    b, a = butter(2, np.array([high_pass, low_pass]) / (0.5 * sampling_rate), 'bandpass')
  elif low_pass:
    b, a = butter(2, low_pass / (0.5 * sampling_rate), 'low')
  elif high_pass:
    b, a = butter(2, high_pass, btype='high', fs=sampling_rate)
  if signal_name in ['filt']:
    if low_pass or high_pass:
      ADC1_f = filtfilt(b, a, ADC1_denoised, padtype='even')
      ADC2_f = filtfilt(b, a, ADC2_denoised, padtype='even')
    else:
      ADC1_f = ADC2_f = None

  if signal_name in ['als']:
    # Remove bleaching using asymmetrical least squares smoothing
    ADC1_f = ADC1_denoised - als(ADC1_denoised)
    ADC2_f = ADC2_denoised - als(ADC2_denoised)

  if signal_name is 'denoised':
    ADC1_f = ADC1_denoised
    ADC2_f = ADC2_denoised

  # lowpass detrended signal
  b, a = butter(2, low_pass / (0.5 * sampling_rate), 'low')
  ADC1_f = filtfilt(b, a, ADC1_f, padtype='even')
  ADC2_f = filtfilt(b, a, ADC2_f, padtype='even')

  return {'subject_ID': subject_ID,
          'region': region,
          'hemisphere': hemisphere,
          'datetime': date_time,
          'datetime_str': date_time.strftime('%Y-%m-%d %H:%M:%S'),
          'mode': mode,
          'sampling_rate': sampling_rate,
          'volts_per_div': volts_per_division,
          'ADC1': ADC1,
          'ADC2': ADC2,
          'ADC1_f': ADC1_f,
          'ADC2_f': ADC2_f,
          'DI1': DI1,
          'DI2': DI2,
          't': t}

def signal_correction(ADC1_denoised, ADC2_denoised, sampling_rate, low_pass=20, high_pass=0.001, return_all_param=False):
  # Motion correction using a bandpass filter
  b, a = butter(2, [high_pass, low_pass], btype='bandpass', fs=sampling_rate)
  GCaMP_motionband = filtfilt(b, a, ADC1_denoised, padtype='even')
  TdTom_motionband = filtfilt(b, a, ADC2_denoised, padtype='even')
  slope, intercept, r_value, p_value, std_err = linregress(x=TdTom_motionband, y=GCaMP_motionband)
  GCaMP_est_motion = intercept + slope * ADC2_denoised
  GCaMP_corrected = ADC1_denoised - GCaMP_est_motion

  #Use a double exponential fit to correct for bleaching
  try:
    popt, pcov = curve_fit(double_exp, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(1, 1e-6, 1, 1e-6))
    fit_line = double_exp(np.arange(len(GCaMP_corrected)), *popt)
  except RuntimeError:
    try:
      popt, pcov = curve_fit(double_exp, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(1, 1e-6, 0, 1e-6))
      fit_line = double_exp(np.arange(len(GCaMP_corrected)), *popt)
    except RuntimeError:
      try:
        popt, pcov = curve_fit(double_exp, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(-1, 1e-6, 0, 1e-6))
        fit_line = double_exp(np.arange(len(GCaMP_corrected)), *popt)
      except RuntimeError:
        popt, pcov = curve_fit(linear_func, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(1e-6, 0))
        fit_line = linear_func(np.arange(len(GCaMP_corrected)), *popt)
        print('LINEAR FIT')
  GCaMP_corrected = GCaMP_corrected - fit_line

  if return_all_param:
    return GCaMP_motionband, TdTom_motionband, slope, intercept, r_value, p_value, std_err, GCaMP_est_motion, \
            popt, pcov, fit_line, GCaMP_corrected
  else:
    return GCaMP_corrected

def sync_photometry_data(photometry_filepath, subject_format_photometry, session, low_pass=5,
                         high_pass=0.001, old_format=False, plot=False):
  # Adapted from https: // pyphotometry.readthedocs.io / en / latest / user - guide / importing - data /
  # Import photometry data.
  photo_data = import_data(photometry_filepath,
                              subject_format=subject_format_photometry,
                              low_pass=low_pass,
                              high_pass=[],  # Set high_pass to False to see bleaching otherwise 0.01
                              old_format=old_format,
                              signal_name='denoised')
  # Correct for motion artifacts and photobleaching ('ADC1_f': low_pass denoised signal)
  print('{} {}'.format(photo_data['subject_ID'], photo_data['datetime_str']))
  GCaMP_corrected = signal_correction(photo_data['ADC1_f'], photo_data['ADC2_f'],
                                         photo_data['sampling_rate'], low_pass=low_pass, high_pass=high_pass)
  # Setup synchronisation
  sync_signal = photo_data['DI2'].astype(int)  # Signal from digital input 2 which has sync pulses.
  pulse_times_pho = (1 + np.where(np.diff(sync_signal) == 1)[0]  # Photometry sync pulse times (ms).
                     * 1000 / photo_data['sampling_rate'])
  if old_format:
    pulse_times_pyc = session.times['Rsync']  # pyControl sync pulse times (ms).
  else:
    pulse_times_pyc = session.times['rsync']  # pyControl sync pulse times (ms).
  aligner = rs.Rsync_aligner(pulse_times_A=pulse_times_pyc, pulse_times_B=pulse_times_pho, plot=plot)

  # Convert photometry sample times into pyControl reference frame.
  sample_times_pho = photo_data['t'] * 1000  # Time of photometry samples in photomery reference frame (ms).
  sample_times_pyc = aligner.B_to_A(sample_times_pho)  # Time of photometry samples in pyControl reference frame (ms).

  # remove the denoised signal before returning the dictionary
  del photo_data['ADC1_f']
  del photo_data['ADC2_f']
  return photo_data, sample_times_pho, sample_times_pyc, GCaMP_corrected

def session_information_from_photo_data(photo_data, keys='long'):
  # reduced all_photo_data variable, removed keys with photometry signals, only kept information about the session
  if keys == 'long':
    keys = ['subject_ID', 'region', 'hemisphere', 'datetime', 'datetime_str', 'sampling_rate', 'ADC1', 'ADC2']
  else:
    keys = ['subject_ID', 'region', 'hemisphere', 'datetime', 'datetime_str', 'sampling_rate']
  all_photo_data_info = [{x:photo_data[i][x] for x in keys} for i in range(len(photo_data))]
  return all_photo_data_info

