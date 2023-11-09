from pyControl.utility import *
import hardware_definition as hw

#-------------------------------------------------------------------------
# States and events.
#-------------------------------------------------------------------------

states= ['init_trial',
         'choice_state',
         'choose_left',
         'choose_right',
         'cue_up_state',
         'cue_down_state',
         'up_state',
         'down_state',
         'choose_up',
         'choose_down',
         'reward_cue',
         'no_reward_cue',
         'reward_up',
         'reward_down',
         'time_out',
         'reward_consumption',
         'inter_trial_interval']

events = ['poke_4', 'poke_4_out',
          'poke_6', 'poke_6_out',
          'poke_5', 'poke_5_out',
          'poke_1', 'poke_1_out',
          'poke_9', 'poke_9_out',
          'end_consumption','session_timer']
          # 'rsync']

initial_state = 'init_trial'

#-------------------------------------------------------------------------
# Variables
#-------------------------------------------------------------------------

v.n_trials = 0
v.n_rewards = 0
v.n_blocks = 0
v.n_block_trials = 0
v.choice = None
v.second_step = None
v.upcoming_blocks = []

v.current_choice_type = None
v.click_volume = 20
v.tone_volume  = 10
v.noise_volume = 25
v.freq_tone = [5000, 12000]

#-------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------

v.stage = 4.5
v.session_duration = 1 * hour
v.delay_choice = 200 * ms
v.delay_cue = 200 * ms
v.delay_reward = 200 * ms
v.iti = [2,4] # Inter trial interval [min, max] (seconds)
v.time_out_len = 2 * second
v.reward_cue_len = 500 * ms
v.reward_len = [100, 100]
v.stop_reward_consumption_dur = 250 * ms
v.cue_dur = 1 * second

v.common_transition = 0
v.rare_transition = 0
v.common_reward = 0
v.rare_reward = 0
v.neutral_reward = 0
v.allowed_choice_type = [] #Force choice
v.trials_block_length = [0,0] # [min, max]
v.print_CA = None
v.choice_type = []
v.fixed_transitions = [True, 'A']
v.allowed_transitions = {'UA': ('NA', 'DA'),
                                     'NA': ('UA', 'DA'),
                                     'DA': ('UA', 'NA')}
v.current_block = []
v.store_block = []
v.start_block = True

def select_transitions():
    if v.fixed_transitions[0] == True:
        if v.fixed_transitions[1] == 'A':
            v.allowed_transitions = {'UA': ('NA', 'DA'),
                                     'NA': ('UA', 'DA'),
                                     'DA': ('UA', 'NA')}
        elif v.fixed_transitions[1] == 'B':
            v.allowed_transitions = {'UB': ('NB', 'DB'),
                                     'NB': ('UB', 'DB'),
                                     'DB': ('UB', 'NB')}
    else:
        v.allowed_transitions = {'UA': ('NA', 'NB', 'DA', 'UB'),
                                 'NA': ('UA', 'DA'),
                                 'DA': ('UA','NA','NB','DB'),
                                 'UB': ('NA', 'NB', 'DB', 'UA'),
                                 'NB': ('UB', 'DB'),
                                 'DB': ('UB', 'NB', 'NA', 'DA')}


def select_stage():
    if v.stage == 1.1:
        '''
            go up or down, rewarded 100% 
        '''
        v.tone_volume = 0
        v.allowed_transitions = ['U', 'U', 'U', 'D', 'D', 'D']
        v.transition_type = sample_without_replacement(v.allowed_transitions)

    elif v.stage == 1.2:
        '''
            go up or down, rewarded 100% 
        '''
        v.allowed_transitions = ['U', 'U', 'U', 'D', 'D', 'D']
        v.transition_type = sample_without_replacement(v.allowed_transitions)

    elif v.stage == 2:
        '''
            make available left and right pokes
        '''
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 1
        v.rare_reward = 1
        v.neutral_reward = 1
        v.allowed_choice_type = ['L', 'R'] #Force choice
        v.trials_block_length = [20,30] # [min, max]
        v.print_CA = None
        v.choice_type = sample_without_replacement(v.allowed_choice_type)

    elif v.stage == 3:
        '''
            make available centre poke
        '''
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 1
        v.rare_reward = 1
        v.neutral_reward = 1
        v.allowed_choice_type = ['L', 'R'] #Force choice
        v.trials_block_length = [20,30] # [min, max]
        v.print_CA = None
        v.choice_type = sample_without_replacement(v.allowed_choice_type)

    elif v.stage == 4.1:
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 0.9
        v.rare_reward = 0.7
        v.neutral_reward = 0.8
        v.allowed_choice_type = ['L', 'R'] #Force choice
        v.trials_block_length = [20,30] # [min, max]
        v.print_CA = None
        v.choice_type = sample_without_replacement(v.allowed_choice_type)

    elif v.stage == 4.2:
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 0.9
        v.rare_reward = 0.5
        v.neutral_reward = 0.7
        v.allowed_choice_type = ['L', 'R', 'L', 'R','L', 'R','FC', 'FC'] #25% free choice
        v.trials_block_length = [20,30] # [min, max]
        v.print_CA = None
        v.choice_type = sample_without_replacement(v.allowed_choice_type)

    elif v.stage == 4.3:
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 0.9
        v.rare_reward = 0.3
        v.neutral_reward = 0.6
        v.allowed_choice_type = ['L', 'R', 'L', 'R','L', 'R','FC', 'FC'] #25% free choice
        v.trials_block_length = [20,30] # [min, max]
        v.print_CA = None
        v.choice_type = sample_without_replacement(v.allowed_choice_type)

    elif v.stage == 4.4:
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 0.9
        v.rare_reward = 0.1
        v.neutral_reward = 0.5
        v.allowed_choice_type = ['L', 'R', 'L', 'R','L', 'R','FC', 'FC'] #25% free choice
        v.trials_block_length = [20,30] # [min, max]
        v.print_CA = None
        v.choice_type = sample_without_replacement(v.allowed_choice_type)

    elif v.stage == 4.5:
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 0.9
        v.rare_reward = 0.1
        v.neutral_reward = 0.5
        v.allowed_choice_type = ['L', 'R', 'FC', 'FC'] #50% free choice
        v.neutral_block_length = [20,30] # [min, max]
        v.trials_post_threshold = [5,15] # [min, max]
        v.mov_ave_correct = exp_mov_ave(tau=8, init_value=0.5)
        v.choice_type = sample_without_replacement(v.allowed_choice_type)

    elif v.stage == 4.6:
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 0.9
        v.rare_reward = 0.1
        v.neutral_reward = 0.5
        v.allowed_choice_type = ['L', 'R', 'FC', 'FC','FC', 'FC','FC', 'FC'] #75% free choice
        v.neutral_block_length = [20,30] # [min, max]
        v.trials_post_threshold = [5,15] # [min, max]
        v.mov_ave_correct = exp_mov_ave(tau=8, init_value=0.5)
        v.choice_type = sample_without_replacement(v.allowed_choice_type)
        
    elif v.stage == 4.7:
        
        v.common_transition = 0.8
        v.rare_transition = 0.2
        v.common_reward = 0.8
        v.rare_reward = 0.2
        v.neutral_reward = 0.5
        v.allowed_choice_type = ['L', 'R', 'FC', 'FC','FC', 'FC','FC', 'FC'] #75% free choice
        v.neutral_block_length = [20,30] # [min, max]
        v.trials_post_threshold = [5,15] # [min, max]
        v.mov_ave_correct = exp_mov_ave(tau=8, init_value=0.5)
        v.choice_type = sample_without_replacement(v.allowed_choice_type)
        

#-------------------------------------------------------------------------
# Define behaviour
#-------------------------------------------------------------------------

def run_start():
    if v.stage >= 2:
        select_transitions()
        select_stage()
        block_transition()
        v.start_block = False
    else:
        select_stage()
    set_timer('session_timer', v.session_duration)
        
def run_end():
    hw.off()

def init_trial(event):
    if event == 'entry':
        hw.speaker.set_volume(v.click_volume)
        if v.stage < 2:
            v.current_transition = v.transition_type.next()
            if v.current_transition is 'U':
                timed_goto_state('up_state', v.delay_choice)
            elif v.current_transition is 'D':
                timed_goto_state('down_state', v.delay_choice)
        elif v.stage == 2:
            v.current_choice_type = v.choice_type.next()
            timed_goto_state('choice_state', v.delay_choice)
        else:
            hw.poke_5.LED.on()
            v.current_choice_type = v.choice_type.next()
    elif event == 'exit':
        hw.poke_5.LED.off()
    elif event == 'poke_5':
        hw.speaker.click()
        goto_state('choice_state')

def choice_state(event):
    if event == 'entry':
        if v.current_choice_type is 'FC':
            hw.poke_4.LED.on()
            hw.poke_6.LED.on()
        elif v.current_choice_type is 'L':
            hw.poke_4.LED.on()
        elif v.current_choice_type is 'R':
            hw.poke_6.LED.on()
    elif event == 'exit':
        hw.poke_4.LED.off()
        hw.poke_6.LED.off()
    elif ((event == 'poke_4') and (v.current_choice_type in {'FC', 'L'})):
        hw.speaker.click()
        goto_state('choose_left')
    elif ((event == 'poke_6') and (v.current_choice_type in {'FC', 'R'})):
        hw.speaker.click()
        goto_state('choose_right')

# Delay to second state
def choose_left(event):
    if event == 'entry':
        v.choice = 'left'
        if (((v.trans_block == 'A') and withprob(v.common_transition)) or
            ((v.trans_block == 'B') and withprob(v.rare_transition))):
            timed_goto_state('cue_up_state', v.delay_choice)
        else:
            timed_goto_state('cue_down_state', v.delay_choice)

#Delay to second state
def choose_right(event):
    if event == 'entry':
        v.choice = 'right'
        if (((v.trans_block == 'A') and withprob(v.common_transition)) or
            ((v.trans_block == 'B') and withprob(v.rare_transition))):
            timed_goto_state('cue_down_state', v.delay_choice)
        else:
            timed_goto_state('cue_up_state', v.delay_choice)

def cue_up_state(event):
    if event == 'entry':
        hw.poke_1.LED.on()
        hw.speaker.sine(v.freq_tone[1])
        timed_goto_state('up_state', v.cue_dur)
    elif event == 'exit':
        hw.speaker.off()

def cue_down_state(event):
    if event == 'entry':
        hw.poke_9.LED.on()
        hw.speaker.sine(v.freq_tone[0])
        timed_goto_state('down_state', v.cue_dur)
    elif event == 'exit':
        hw.speaker.off()

def up_state(event):
    if event == 'entry':
        hw.poke_1.LED.on()
    elif event == 'exit':
        hw.poke_1.LED.off()
    elif event == 'poke_1':
        hw.speaker.click()
        goto_state('choose_up')

def down_state(event):
    if event == 'entry':
        hw.poke_9.LED.on()
    elif event == 'exit':
        hw.poke_9.LED.off()
    elif event == 'poke_9':
        hw.speaker.click()
        goto_state('choose_down')

#Delay to cue
def choose_up(event):
    if event == 'entry':
        v.second_step = 'up'
        if v.stage < 2:
            timed_goto_state('reward_cue', v.delay_cue)
            v.outcome = 1
            v.n_rewards += 1
        else:    
            if (((v.reward_block == 'D') and withprob(v.rare_reward)) or
                ((v.reward_block == 'N') and withprob(v.neutral_reward)) or
                ((v.reward_block == 'U') and withprob(v.common_reward))):
                timed_goto_state('reward_cue', v.delay_cue)
                v.outcome = 1
                v.n_rewards += 1 

            elif (((v.reward_block == 'D') and withprob(v.common_reward)) or
                  ((v.reward_block == 'N') and withprob(v.neutral_reward)) or
                  ((v.reward_block == 'U') and withprob(v.rare_reward))):
                  timed_goto_state('no_reward_cue', v.delay_cue)
                  v.outcome = 0
            else:
                timed_goto_state('no_reward_cue', v.delay_cue)
                v.outcome = 0

#Delay to cue
def choose_down(event):
    if event == 'entry':
        v.second_step = 'down'
        if v.stage < 2:
            timed_goto_state('reward_cue', v.delay_cue)
            v.outcome = 1
            v.n_rewards += 1
        else:    
            if (((v.reward_block == 'D') and withprob(v.common_reward)) or
                ((v.reward_block == 'N') and withprob(v.neutral_reward)) or
                ((v.reward_block == 'U') and withprob(v.rare_reward))):
                timed_goto_state('reward_cue', v.delay_cue)
                v.outcome = 1
                v.n_rewards += 1

            elif (((v.reward_block == 'D') and withprob(v.rare_reward))  or
                  ((v.reward_block == 'N') and withprob(v.neutral_reward)) or
                  ((v.reward_block == 'U') and withprob(v.common_reward))):
                   timed_goto_state('no_reward_cue', v.delay_cue)
                   v.outcome = 0
            else:
                timed_goto_state('no_reward_cue', v.delay_cue)
                v.outcome = 0

def reward_cue(event):
    if event == 'entry':
        hw.speaker.set_volume(v.tone_volume)
        if v.second_step == 'up':
            hw.speaker.pulsed_sine(v.freq_tone[1],10)
            timed_goto_state('reward_up', v.reward_cue_len)
        elif v.second_step == 'down':
            hw.speaker.pulsed_sine(v.freq_tone[0],10)
            timed_goto_state('reward_down', v.reward_cue_len)
    elif event == 'exit':
        hw.speaker.off()

def no_reward_cue(event):
    if event == 'entry':
        hw.speaker.set_volume(v.noise_volume)
        timed_goto_state('time_out', v.reward_cue_len)
        hw.speaker.noise()
    elif event == 'exit':
        hw.speaker.off()

def time_out(event):
    if event == 'entry':
        timed_goto_state('reward_consumption', v.time_out_len)

def reward_up(event):
    if event == 'entry':
        hw.poke_1.SOL.on()
        # hw.BNC_1.on()
        timed_goto_state('reward_consumption', v.reward_len[0] * ms)
    elif event == 'exit':
        hw.poke_1.SOL.off()
        # hw.BNC_1.off()

def reward_down(event):
    if event == 'entry':
        hw.poke_9.SOL.on()
        # hw.BNC_1.on()
        timed_goto_state('reward_consumption', v.reward_len[1] * ms)
        
    elif event == 'exit':
        hw.poke_9.SOL.off()
        # hw.BNC_1.off()

def reward_consumption(event):
    if event == 'entry':
        if not (hw.poke_1.value() or hw.poke_9.value()): # subject already left poke.
            set_timer('end_consumption', v.stop_reward_consumption_dur)
    elif event in ('poke_1', 'poke_9'):
        disarm_timer('end_consumption')
    elif event in ('poke_1_out', 'poke_9_out'):
        set_timer('end_consumption', v.stop_reward_consumption_dur)
    elif event == 'end_consumption':
        trial_update()
        # timed_goto_state('init_trial', v.iti)
        goto_state('inter_trial_interval')

# ITI time
def inter_trial_interval(event):
    if event == 'entry':
        timed_goto_state('init_trial', randint(*v.iti)*second)

def all_states(event):
    if event == 'session_timer':
        stop_framework()

#-------------------------------------------------------------------------
# Block structure code
#-------------------------------------------------------------------------

def trial_update():
    # Update correct moving average
    v.n_block_trials += 1
    v.n_trials += 1
    if v.stage >= 4.5:
        if v.reward_block is 'N':
            v.mov_ave_correct.update(0.5)
        elif v.current_choice_type is 'FC': #Update moving average just if it was a free choice trial
            if v.choice == v.correct_trial:
                v.mov_ave_correct.update(1)
            else:
                v.mov_ave_correct.update(0)
        v.print_CA = v.mov_ave_correct.value
    #Print parameters
    if v.choice is 'left':
        choice_print = 1
    else:
        choice_print = 0
    if v.second_step is 'up':
        second_print = 1
    else:
        second_print = 0
    print('T#:{} R#:{} B#:{} C:{} S:{} O:{} CA:{} B:{} CT:{} TS:{}'.format(v.n_trials,
    v.n_rewards, v.n_blocks, choice_print, second_print, v.outcome,
    v.print_CA, v.current_block, v.current_choice_type, v.stage))
    # Check if change block type
    if v.stage >= 4.5:
        if v.reward_block is 'N':
            if v.n_block_trials >= v.block_length:
                block_transition()
        else:
            if v.mov_ave_correct.value > 0.75:
                if v.n_post_thres_trials >= v.block_length:
                    block_transition()
                else:
                    v.n_post_thres_trials += 1
    elif v.stage >= 2:
        if v.n_block_trials >= v.block_length:
            block_transition()

def block_transition():
    if v.start_block == False:
        next_block = None
        # if v.upcoming_blocks:
        try:
            next_block = next(b for b in v.upcoming_blocks if b in v.allowed_transitions[v.current_block]) #get the first block that is allowed
        # if not next_block: #if there is no block available, add another set of blocks.
        except StopIteration:
            v.upcoming_blocks += shuffled(v.allowed_transitions.keys())
            next_block = next(b for b in v.upcoming_blocks if b in v.allowed_transitions[v.current_block])
        del v.upcoming_blocks[v.upcoming_blocks.index(next_block)]
        v.current_block = next_block
    else:
        if not v.store_block:
            v.current_block = choice(list(v.allowed_transitions.keys()))
        else:
            v.current_block = v.store_block[0]
        next_block = v.current_block
    v.reward_block = v.current_block[0]
    v.trans_block = v.current_block[1]
    if v.stage >= 4.5:
        v.mov_ave_correct.value = 1 - v.mov_ave_correct.value
        if next_block in ('UA', 'DB'):
            v.correct_trial = 'left'
            v.block_length = randint(*v.trials_post_threshold)
        elif next_block in ('DA', 'UB'):
            v.correct_trial = 'right'
            v.block_length = randint(*v.trials_post_threshold)
        else:
            v.correct_trial = 'neutral'
            v.mov_ave_correct.value = 0.5
            v.block_length = randint(*v.neutral_block_length)
        v.n_blocks += 1
        v.n_block_trials = 0
        v.n_post_thres_trials = 0
    else:
        v.block_length = randint(*v.trials_block_length)
        v.n_blocks += 1
        v.n_block_trials = 0
    v.store_block = [v.current_block]