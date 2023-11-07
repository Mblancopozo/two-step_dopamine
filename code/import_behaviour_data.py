#----------------------------------------------------------------------------------
# Code to import behavioural data
# Marta Blanco-Pozo, 2023

# Adapted from https://github.com/pyControl/code/blob/master/tools/data_import.py
#----------------------------------------------------------------------------------

import sys, os
import pickle
import numpy as np
from datetime import datetime, date
from collections import namedtuple
import re
import more_itertools as mit

Event = namedtuple('Event', ['time','name'])

data_sets_path = '..\\data sets'

#----------------------------------------------------------------------------------
# Session class
#----------------------------------------------------------------------------------
class Session():
    '''Import data from a pyControl file and represent it as an object with attributes:
      - file_name
      - experiment_name
      - task_name
      - subject_ID
          If argument int_subject_IDs is True, suject_ID is stored as an integer,
          otherwise subject_ID is stored as a string.
      - datetime
          The date and time that the session started stored as a datetime object.
      - datetime_string
          The date and time that the session started stored as a string of format 'YYYY-MM-DD HH:MM:SS'
      - events
          A list of all framework events and state entries in the order they occured.
          Each entry is a namedtuple with fields 'time' & 'name', such that you can get the
          name and time of event/state entry x with x.name and x.time respectively.
      - times
          A dictionary with keys that are the names of the framework events and states and
          corresponding values which are Numpy arrays of all the times (in milliseconds since the
           start of the framework run) at which each event/state entry occured.
      - print_lines
          A list of all the lines output by print statements during the framework run, each line starts
          with the time in milliseconds at which it was printed.
    '''

    def __init__(self, file_path, int_subject_IDs=True):

        # Load lines from file.

        with open(file_path, 'r') as f:
            print('Importing data file: '+os.path.split(file_path)[1])
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Extract and store session information.

        self.file_name = os.path.split(file_path)[1]

        info_lines = [line[2:] for line in all_lines if line[0]=='I']

        self.experiment_name = next(line for line in info_lines if 'Experiment name' in line).split(' : ')[1]
        self.task_name       = next(line for line in info_lines if 'Task name'       in line).split(' : ')[1]
        subject_ID_string    = next(line for line in info_lines if 'Subject ID'      in line).split(' : ')[1]
        datetime_string      = next(line for line in info_lines if 'Start date'      in line).split(' : ')[1]

        if int_subject_IDs: # Convert subject ID string to integer.
            self.subject_ID = int(''.join([i for i in subject_ID_string if i.isdigit()]))
        else:
            self.subject_ID = subject_ID_string

        self.datetime = datetime.strptime(datetime_string, '%Y/%m/%d %H:%M:%S')
        self.datetime_string = self.datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Extract and store session data.

        state_IDs = eval(next(line for line in all_lines if line[0]=='S')[2:])
        event_IDs = eval(next(line for line in all_lines if line[0]=='E')[2:])

        ID2name = {v: k for k, v in {**state_IDs, **event_IDs}.items()}

        data_lines = [line[2:].split(' ') for line in all_lines if line[0]=='D']
        data_print_lines = [line[2:].split(' ') for line in all_lines if line[0] in ['D', 'P']]

        self.events = [Event(int(dl[0]), ID2name[int(dl[1])]) for dl in data_lines]

        self.times = {event_name: np.array([ev.time for ev in self.events if ev.name == event_name])
                      for event_name in ID2name.values()}

        self.events_and_print = [Event(int(dl[0]), ID2name[int(dl[1])]) if dl[1].isdigit() else
                                 Event(int(dl[0]), dl[1:]) for dl in data_print_lines]

        self.print_lines = [line[2:] for line in all_lines if line[0]=='P']

        print_lines = [line[2:].split(' ',1) for line in all_lines if line[0]=='P']

        # -------------------------------------------------------------------------------------------
        # Make dictionary of choices, transitions, second steps and outcomes on each trial.
        #--------------------------------------------------------------------------------------------

        trial_lines = [line[1] for line in print_lines if line[1][:2] == 'T#'] # Lines with trial data.

        if (self.experiment_name == 'Five_poke_two_step') or (self.experiment_name == 'run_task') or \
                (self.experiment_name == 'Two_step_fixed') or (self.experiment_name == 'Five_poke_two_step_A') or \
                ('Two_step' in self.experiment_name) or ('two_step' in self.experiment_name):

            # Example trial line: '2494174 T#:77 R#:52 B#:4 C:1 S:1 O:0 CA:None B:UA CT:FC TS:4.2'

            choices      = np.array([int(tl[tl.find('C:')+2]) for tl in trial_lines]) # 1: left,    0: right
            second_steps = np.array([int(tl[tl.find('S:')+2]) for tl in trial_lines]) # 1: up,      0: down
            outcomes     = np.array([int(tl[tl.find('O:')+2]) for tl in trial_lines]) # 1: reward   0: no reward
            transitions  = (choices == second_steps).astype(int)
            free_choice  = np.array([tl[tl.find('CT:')+3] == 'F'
                           for tl in trial_lines])                                       # True: Free choice, False: Forced choice
            if 'TS' in trial_lines[0]:
                stage        = np.array([float(re.search('TS:(\d)', tl).group(1)) for tl in trial_lines])
                if stage[0] >= 4:
                  stage        = np.array([float(re.search('TS:(\d+\.\d+)', tl).group(1)) for tl in trial_lines])
                if stage[0] >= 4.5:
                    mov_average  = [float(re.search('CA:(\d+\.\d+)', tl).group(1)) for tl in trial_lines]
                else:
                    mov_average = np.array([tl[tl.find('CA:')+3] for tl in trial_lines])
            else:
                stage = 'NaN'
                mov_average = 'NaN'

            if 'RL' in trial_lines[0]:
                self.reward_loc = re.search('RL:(\w\w)', trial_lines[0]).group(1)
            else:
                self.reward_loc = 'UD'

            if 'stim' in trial_lines[0]:
                stim = np.array([int(tl[tl.find('stim:')+5]) for tl in trial_lines if 'stim' in tl])
                stim_type = [tl.split()[11].split(':')[-1] for tl in trial_lines]
                self.n_stim = sum(stim)
            else:
                stim = 'NaN'
                stim_type = 'NaN'
                self.n_stim = 'NaN'


            self.trial_data = {'choices'     : choices,      'second_steps': second_steps,
                               'outcomes'    : outcomes,     'transitions' : transitions,
                               'free_choice' : free_choice,  'stage'       : stage,
                               'mov_average' : mov_average,  'stim'        : stim,
                               'stim_type'   : stim_type}

            self.n_trials = len(choices)
            self.rewards  = sum(outcomes)
            self.fraction_rewarded = self.rewards/self.n_trials


            # --------------------------------------------------------------------------------------------
            # Extract block information.
            # --------------------------------------------------------------------------------------------

            if self.reward_loc == 'UD':
                trial_rew_state  =  np.array([(int(tl[tl.find('B:')+2] == 'U') or 2 * int(tl[tl.find('B:')+2] == 'N'))
                                for tl in trial_lines]) # Reward state for each trial.
            else:
                trial_rew_state = np.array([(int(tl[tl.find('B:') + 2] == 'L') or 2 * int(tl[tl.find('B:') + 2] == 'N'))
                                            for tl in trial_lines])  # Reward state for each trial.
                                                        # 1: left, 0: right, 2: neutral
            trial_trans_state = np.array([bool(int(tl[tl.find('B:')+3] == 'A')) for tl in trial_lines])  # Transition state for each trial.
                                                                                                         # True: A, False: B
            block_start_mask = ((trial_rew_state[1:] != trial_rew_state[:-1]) |
                                (trial_trans_state[1:] !=  trial_trans_state[:-1]))
            start_trials = [0] + (np.where(block_start_mask)[0] + 1).astype(int).tolist() # Block start trials (trials numbered from 0).
            transition_states = trial_trans_state[np.array(start_trials)].astype(int)     # Transition state for each block.
            reward_states = trial_rew_state[np.array(start_trials)]                       # Reward state for each block.

            self.blocks = {'start_trials'      : start_trials,
                           'end_trials'        : start_trials[1:] + [self.n_trials],
                           'start_times'       : None,
                           'reward_states'     : reward_states,      # 1 for up good, 2 for neutral, 0 for down good.
                           'transition_states' : transition_states,  # 1 for A blocks, 0 for B blocks.
                           'trial_trans_state' : trial_trans_state,
                           'trial_rew_state'   : trial_rew_state}

        #--------------------------------------------------------------------------------------------

        elif 'self_stim' in self.experiment_name:
            variables = [line[2:].split(' ') for line in all_lines if line[0] == 'V']
            reversal = [v[2] for v in variables if v[0] == '-1' and v[1] == 'reversal']
            extinction = [v[2] for v in variables if v[0] == '-1' and v[1] == 'extinction']

            stim_param = [line[1] for line in print_lines if line[1][:15] == 'Stim parameters']
            block_lines = [line[1] for line in print_lines if line[1][:13] == 'Block summary']
            start_block_lines =[line[1] for line in print_lines if line[1][:6] == 'Block:']

            freq = [param.split(',')[1].split()[-1] for param in stim_param]
            duty_cycle = [param.split(',')[2].split(':')[-1] for param in stim_param]
            pulses = [param.split(',')[3].split()[-1] for param in stim_param]
            min_ISI = [param.split(',')[4].split()[-1] for param in stim_param]

            start_block_time = [int(line[0]) for line in print_lines if line[1][:6] == 'Block:']
            end_block_time = [int(line[0]) for line in print_lines if line[1][:13] == 'Block summary']
            # active_poke = [int(re.findall(r'%s(\d+)' % 'AP:', bl)[0]) for bl in block_lines]
            active_poke = [bl.split()[-1] for bl in start_block_lines]
            n_stim = [int(re.findall(r'%s(\d+)' % 'NS:', bl)[0]) for bl in block_lines]
            n_active = [int(re.findall(r'%s(\d+)' % 'NA:', bl)[0]) for bl in block_lines]
            n_inactive = [int(re.findall(r'%s(\d+)' % 'NI:', bl)[0]) for bl in block_lines]
            try:
                if re.findall(r'%s([A-Za-z])' % 'AP:', block_lines[-1])[0] == 'e':
                    # the last block was an extinction block
                    start_block_time += [end_block_time[-2]]
                    active_poke += ['None']
                    n_stim = [n_stim[0]] + [y-x for x, y in zip(n_stim, n_stim[1:])]
                    n_active = [n_active[0]] + [y-x for x, y in zip(n_active, n_active[1:])]
                    n_inactive = [n_inactive[0]] + [y-x for x, y in zip(n_inactive, n_inactive[1:])]
            except:
                print('This session does not have extinction at the end')


            start_pos = [max(list(group)) for group in mit.consecutive_groups(np.where([self.events_and_print[i].time
                                                                            in start_block_time for i in range(len(self.events_and_print))])[0])]
            end_pos = np.append(start_pos[1:], len(self.events_and_print))
            if self.events_and_print[-2].name[0] == 'Block':
                add = 2
            else:
                add = 1
            block_events = [[self.events_and_print[x].name for x in np.arange(start_pos[i] + 1, int(end_pos[i]) - add)]
                            for i in range(len(start_pos))]
            dict_keys = ['poke_2', 'poke_3', 'poke_7', 'poke_8']
            dict_poke_count = {k: [block_events[i].count(k) for i in range(len(block_events))] for k in dict_keys}

            self.variables = {'reversal': reversal, 'extiction': extinction}
            self.stim_param = {'freq': freq, 'duty_cycle': duty_cycle, 'pulses': pulses, 'min_ISI': min_ISI}

            self.block_data = {'start_block_time': start_block_time, 'end_block_time': end_block_time,
                               'active_poke': active_poke, 'n_stim': n_stim, 'n_active': n_active,
                               'n_inactive': n_inactive, 'poke_counts': dict_poke_count}

    def select_trials(self, selection_type, select_n = 20, first_n_mins = False,
                  block_type = 'all'):
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

        if selection_type == 'xtr': # Select all trials except select_n following transition reversal only
            trials_to_use = np.ones(self.n_trials, dtype = bool)
            trans_change = np.hstack((
                False, ~np.equal(self.blocks['transition_states'][:-1],
                                 self.blocks['transition_states'][1:])))
            start_trials = (self.blocks['start_trials'] +
                            [self.blocks['end_trials'][-1] + select_n])
            for i in range(len(trans_change)):
                if trans_change[i]:
                    trials_to_use[start_trials[i]:start_trials[i] + select_n] = False
        if selection_type == 'xtrrw': # Select all trials except select_n following block reversal
            trials_to_use = np.ones(self.n_trials, dtype = bool)
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

        elif selection_type == 'xmid': #select trials in the middle
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

        elif selection_type == 'end': # Select only select_n trials before block transitions.
            trials_to_use = np.zeros(self.n_trials, dtype = bool)
            for b in self.blocks['start_trials'][1:]:
                trials_to_use[b - select_n:b] = True #ELIMINATED -1 FROM ORIGINAL CODE

        elif selection_type == 'start': # Select only select_n trials after block transitions.
            trials_to_use = np.zeros(self.n_trials, dtype = bool)
            for b in self.blocks['start_trials'][:]:
                trials_to_use[b:b + select_n] = True

        elif selection_type == 'start_1': # Select only select_n trials after block transitions but eliminating the first trial.
            trials_to_use = np.zeros(self.n_trials, dtype = bool)
            for b in self.blocks['start_trials'][1:]:
                trials_to_use[b+1:b+1 + select_n] = True

        elif selection_type == 'all': # Use all trials.
            trials_to_use = np.ones(self.n_trials, dtype = bool)

        if first_n_mins:  #  Restrict analysed trials to only first n minutes.
            time_selection = self.times['trial_start'][:self.n_trials] < (60*first_n_mins)
            trials_to_use = trials_to_use & time_selection

        if not block_type == 'all': #  Restrict analysed trials to blocks of certain types.
            if block_type == 'neutral':       # Include trials only from neutral blocks.
                block_selection = self.blocks['trial_rew_state'] == 2
            elif block_type == 'non_neutral': # Include trials only from non-neutral blocks.
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

    def unpack_trial_data(self, order = 'CTSO', dtype = int):
        'Return elements of trial_data dictionary in specified order and data type.'
        o_dict = {'C': 'choices', 'T': 'transitions', 'S': 'second_steps', 'O': 'outcomes'}
        if dtype == int:
            return [self.trial_data[o_dict[i]] for i in order]
        else:
            return [self.trial_data[o_dict[i]].astype(dtype) for i in order]


#----------------------------------------------------------------------------------
# Experiment class
#----------------------------------------------------------------------------------

class Experiment():
    def __init__(self, folder_path, int_subject_IDs=True):
        '''
        Import all sessions from specified folder to create experiment object.  Only sessions in the
        specified folder (not in subfolders) will be imported.
        Arguments:
        folder_path: Path of data folder.
        int_subject_IDs:  If True subject IDs are converted to integers, e.g. m012 is converted to 12.
        '''

        self.folder_name = os.path.split(folder_path)[1]
        self.path = folder_path

        # Import sessions.

        self.sessions = []
        try: # Load sessions from saved sessions.pkl file.
            with open(os.path.join(self.path, 'sessions.pkl'),'rb') as sessions_file:
                self.sessions = pickle.load(sessions_file)
            print('Saved sessions loaded from: sessions.pkl')
        except IOError:
           pass

        old_files = [session.file_name for session in self.sessions]
        files = os.listdir(self.path)
        new_files = [f for f in files if f[-4:] == '.txt' and f not in old_files]

        if len(new_files) > 0:
            print('Loading new data files..')
            for file_name in new_files:
                try:
                    self.sessions.append(Session(os.path.join(self.path, file_name), int_subject_IDs))
                except Exception as error_message:
                    print('Unable to import file: ' + file_name)
                    print(error_message)

        # Assign session numbers.

        self.subject_IDs = list(set([s.subject_ID for s in self.sessions]))
        self.n_subjects = len(self.subject_IDs)

        self.sessions.sort(key = lambda s:s.datetime_string + str(s.subject_ID))

        self.sessions_per_subject = {}
        for subject_ID in self.subject_IDs:
            subject_sessions = self.get_sessions(subject_ID)
            for i, session in enumerate(subject_sessions):
                session.number = i+1
            self.sessions_per_subject[subject_ID] = subject_sessions[-1].number

    def save(self):
        '''Save all sessions as .pkl file. Speeds up subsequent instantiation of
        experiment as sessions do not need to be reimported from data files.'''
        with open(os.path.join(self.path, 'sessions.pkl'),'wb') as sessions_file:
            pickle.dump(self.sessions, sessions_file)

    def get_sessions(self, subject_IDs='all', when='all'):
        '''Return list of sessions which match specified subject ID and time.
        Arguments:
        subject_ID: Set to 'all' to select sessions from all subjects or provide a list of subject IDs.
        when      : Determines session number or dates to select, see example usage below:
                    when = 'all'      # All sessions
                    when = 1          # Sessions numbered 1
                    when = [3,5,8]    # Session numbered 3,5 & 8
                    when = [...,10]   # Sessions numbered <= 10
                    when = [5,...]    # Sessions numbered >= 5
                    when = [5,...,10] # Sessions numbered 5 <= n <= 10
                    when = '2017-07-07' # Select sessions from date '2017-07-07'
                    when = ['2017-07-07','2017-07-08'] # Select specified list of dates
                    when = [...,'2017-07-07'] # Select session with date <= '2017-07-07'
                    when = ['2017-07-01',...,'2017-07-07'] # Select session with '2017-07-01' <= date <= '2017-07-07'.
        '''
        if subject_IDs == 'all':
            subject_IDs = self.subject_IDs
        if not isinstance(subject_IDs, list):
            subject_IDs = [subject_IDs]

        if when == 'all': # Select all sessions.
            when_func = lambda session: True

        else:
            if type(when) is not list:
                when = [when]

            if ... in when: # Select a range..

                if len(when) == 3:  # Start and end points defined.
                    assert type(when[0]) == type(when[2]), 'Start and end of time range must be same type.'
                    if type(when[0]) == int: # .. range of session numbers.
                        when_func = lambda session: when[0] <= session.number <= when[2]
                    else: # .. range of dates.
                        when_func = lambda session: _toDate(when[0]) <= session.datetime.date() <= _toDate(when[2])

                elif when.index(...) == 0: # End point only defined.
                    if type(when[1]) == int: # .. range of session numbers.
                        when_func = lambda session: session.number <= when[1]
                    else: # .. range of dates.
                        when_func = lambda session: session.datetime.date() <= _toDate(when[1])

                else: # Start point only defined.
                    if type(when[0]) == int: # .. range of session numbers.
                        when_func = lambda session: when[0] <= session.number
                    else: # .. range of dates.
                        when_func = lambda session: _toDate(when[0]) <= session.datetime.date()

            else: # Select specified..
                assert all([type(when[0]) == type(w) for w in when]), "All elements of 'when' must be same type."
                if type(when[0]) == int: # .. session numbers.
                    when_func = lambda session: session.number in when
                else: # .. dates.
                    dates = [_toDate(d) for d in when]
                    when_func = lambda session: session.datetime.date() in dates

        valid_sessions = [s for s in self.sessions if s.subject_ID in subject_IDs and when_func(s)]

        return valid_sessions


def _toDate(d): # Convert input to datetime.date object.
    if type(d) is str:
        try:
            return datetime.strptime(d, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError('Unable to convert string to date, format must be YYYY-MM-DD.')
    elif type(d) is datetime:
        return d.date()
    elif type(d) is date:
        return d
    else:
        raise ValueError('Unable to convert input to date.')
