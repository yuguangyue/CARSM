import tensorflow as tf
import numpy as np
import random
import gym
import tensorflow.contrib.eager as tfe
from carsm_util_2 import *
from carsm_util import *
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from baselines.common.cmd_util import make_vec_env
import sys
from baselines import logger
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.schedules import LinearSchedule
from baselines.a2c.utils import discount_with_dones
import os
from trpo_agent import *
from collections import deque

        
tf.enable_eager_execution()
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
################################# set seed ####################################
seedi=0
tf.set_random_seed(seedi)
np.random.seed(seedi)
random.seed(seedi)
###############################################################################

################################# env setting #################################
env_id = 'Swimmer-v2'
num_env = 1
env_type = 'mujoco'
env = make_vec_env(env_id, env_type, num_env, np.random.randint(0, 1e+9), reward_scale=1, flatten_dict_observations=True)
C = 11 # manully select how to split each dimension
action_high = np.float32(env.action_space.high)
action_low = np.float32(env.action_space.low)
gap = np.reshape((action_high - action_low) / (C-1),(1,-1))
K = len(action_high)
nA = K * C # the output of Neural Network
nS = env.observation_space.shape[0]
train_freq = 1
###############################################################################

############################### hyperparameters ###############################
total_timesteps = np.int(1e6) # total frames to run
n_buff = np.int(1e5) # frames in buffer
nstep = 2048 # number of steps of the vectorized environment per update
n_iter_Q = 100
grad_par_input = 1
n_iter_policy = 1
lr_q = 0.01
entropy_par = 0.01
coef_q = 1 # equivalent to change lr_q
gamma = .99 #reward discount factor
Grad_Clip_NN = False
Grad_Clip_Norm = 0.5
tau = 0.01
minibatch = 2048 # size of off-policy samples

damping = 1e-2
KL_par = 1
d_target = 0.01
threshold_input = 0.01
###############################################################################

################################ build agent ##################################
network = 'Linear'
model = ARSM_TRPO_AGENT_mujoco(nA, network, tau, Grad_Clip_Norm, K, C, KL_par, d_target)
logger.configure()
###############################################################################

################################ initialize buffer ############################
replay_buffer = ReplayBuffer(n_buff)
beta_schedule = None
###############################################################################



########### initialize network##############
state = env.reset()
action = np.float32(np.zeros((1, K)))
tt = model.policy(tf.convert_to_tensor(state))[0]
tt = model.dup_policy(tf.convert_to_tensor(state))[0]
tt = model.q_estimation(tf.convert_to_tensor(np.concatenate((state, action), axis=1)))[0]
tt = model.dup_q_estimation(tf.convert_to_tensor(np.concatenate((state, action), axis=1)))[0]
model.duplicate()

timestep,sample_time,update_time = 0,0,0
reward_history,reward_smooth,loss_record = [],[],[]
policy_loss_save, entropy_loss_save, q_loss_save, pg_norm_save, qg_norm_save = [],[],[],[],[]
random_choose = False
score_total = []
timestep_total = []
###############################################################################

d1 = deque(maxlen = 100)

time_tic = time.time()
timestep_tmp = 0

optimizer_q = tf.train.AdamOptimizer(lr_q)


while (timestep < total_timesteps):
    global_step = tfe.Variable(timestep)
    entropy_par = scheduler(entropy_par, total_timesteps, timestep, 0.5)
    

    initial_state = env.reset()
    nsteps = nstep
    ##Sample true action sequence
    rewards, actions, states, dones, phi_sequence, pi_sequence, last_state,actions_idx,last_done =\
    runner_mujoco(env, nsteps, model, gamma, initial_state, nA, env_type, replay_buffer,K, C, action_high, action_low, random_choose, one_hot = False)
    sample_time += 1
    timestep += len(dones)
####################### modify the dataframe dimension ########################
    
    if (sample_time-1) % 10 == 0:
        IsPlot = True
    else:
        IsPlot = False
    
    if (sample_time % train_freq == 0):
        states = np.vstack(states)
        actions_idx = np.vstack(actions_idx)
        actions = np.array(actions)
        
        rewards_tmp = rewards.copy()
        last_value = expected_sarsa(model,last_state,K,C,action_low,action_high,False,random_choose,num=100)
        rewards_tmp.append(last_value)
        Q_target = discount_with_dones(rewards_tmp, dones+[last_done], gamma)
        Q_target = np.float32(np.vstack(Q_target))[:-1]
        
        R_buffer_sample = replay_buffer.sample(np.min([minibatch,timestep]))
        next_states_sampled = np.squeeze(R_buffer_sample[3], axis=1)
        dones_sampled = R_buffer_sample[4]
        reward_sampled = R_buffer_sample[2]
        
        last_v = [expected_sarsa(model,np.reshape(state_tmp,(1,-1)),K,C,action_low,action_high,True,random_choose,num=100) for state_tmp in next_states_sampled]
        last_v = np.vstack(last_v)
        Q_target_hist = reward_sampled + last_v * (1-dones_sampled) * gamma
        
        states_sampled1 = np.squeeze(R_buffer_sample[0], axis=1)
        states_sampled2 = states
        states_sampled = np.concatenate((states_sampled1,states_sampled2), axis = 0)
        actions_sampled1 = R_buffer_sample[1]
        actions_sampled2 = actions
        actions_sampled = np.concatenate((actions_sampled1, actions_sampled2), axis = 0)
        target = np.reshape(np.concatenate((Q_target_hist, Q_target), axis = 0), (-1))
###############################################################################
        loss_q_before = float(model.loss_fn_q(tf.convert_to_tensor(states_sampled), tf.convert_to_tensor(target),\
                                              actions_sampled, coef_q))
        for _ in range(n_iter_Q):
            grad_q = model.update_q(states_sampled, optimizer_q, \
                           tf.convert_to_tensor(target), Grad_Clip_NN, actions_sampled, coef_q) 
        loss_q = float(model.loss_fn_q(tf.convert_to_tensor(states_sampled), tf.convert_to_tensor(target),\
                        actions_sampled, coef_q))


        n_f = len(rewards)
        
        #### parallel part I
        ttt = time.time()
        ncpu = multiprocessing.cpu_count()
        pool = ProcessPoolExecutor(ncpu)
        futures = [pool.submit(parallel_calculate_Q_part1_eliminateOneDim, a,b,c,d,e,C,K,random_choose,gap,action_low) \
                   for a,b,c,d,e in zip(actions_idx,pi_sequence,phi_sequence,states,Q_target)]
        
        input_total = [futures[i].result() for i in range(len(futures))]
        n_pseudo = len(input_total)
        
        Q_save = []
        temp_save = []
        dim_var = []
        pseudo_idx_save = []
        for ii in range(len(futures)):
            temp = input_total[ii][0]
            temp_save.append(temp)
            dim_var.append(input_total[ii][2])
            pseudo_idx_save.append(input_total[ii][1])
            if len(temp) > 0:
                if random_choose:
                    action_tmp = np.float32(gap * temp + np.reshape(action_low, (1,-1)) + np.random.uniform(0,gap[0],size=(temp.shape)))## check
                else:
                    action_tmp = np.float32(gap * temp + np.reshape(action_low, (1,-1)))## check
                action_tmp = np.vstack(action_tmp)
                state_tmp_tile = np.tile(states[ii], (len(temp),1))                
                input_tmp = np.concatenate((state_tmp_tile, action_tmp), axis = 1)
                Q_tmp = model.q_estimation(input_tmp)
                Q_save.append(Q_tmp)
            else:
                Q_save.append([])
        ##### parallel part II
        pool = ProcessPoolExecutor(ncpu)
        futures = [pool.submit(parallel_calculate_Q_part2_eliminateOneDim, a,b,c,d,e,f,C,K) \
                   for a,b,c,d,e,f in zip(Q_save,temp_save,pseudo_idx_save,Q_target,pi_sequence,dim_var)]
        f_save = [futures[i].result() for i in range(len(futures))]
        f_save = np.vstack(f_save)
            
        f_delta = tf.convert_to_tensor(-f_save, dtype=tf.float32)

#################################### Apply TRPO ####################################
        policy_grad = model.grads_fn_policy(states, f_delta, entropy_par, K, C)
        policy_grad_flat = tf.concat([tf.reshape(grad_,[-1])for grad_ in policy_grad if grad_ is not None],axis = 0)
        logits_old = model.policy(states)
        v = np.array(policy_grad_flat)
        
        stepdir = cg_arsm(fisher_vector_product, v, model, states, logits_old, cg_iters=10, verbose=0)
        stepdir = stepdir + damping * policy_grad_flat

        threshold = threshold_input

        shs = float(0.5 * np.reshape(stepdir,(1,-1)) @ np.reshape(fisher_vector_product(stepdir, model, states, logits_old),(-1,1)))
        lm = np.sqrt(shs / threshold)
        
        stepdir /= lm
        grad_dir = set_from_flat(stepdir, policy_grad)

        surrbefore = -float(model.loss_fn_policy(states, f_delta, entropy_par, K, C))
        
        store_1 = copy.deepcopy(model.dense_layer1.variables)
        store_2 = copy.deepcopy(model.dense_layer2.variables)
        store_3 = copy.deepcopy(model.policy_layer.variables)
        
        T = len(states)
        grad_par = grad_par_input
        adj = 0
        while True:
            for kk in range(len(grad_dir)):
                if grad_dir[kk] is not None:
                    tf.assign(model.variables[kk], model.variables[kk] - grad_par * grad_dir[kk])
            KL_divergence = model.KL_div(states, logits_old)  
            surr = -float(model.loss_fn_policy(states, f_delta, entropy_par, K, C))
            if KL_divergence/(T*K) < 1.5 * threshold and surr > surrbefore:
                break
            else:
                grad_par /= 2
                adj+=1
            useless_output = [v_t.assign(v) for v_t, v in zip(model.dense_layer1.variables, store_1)]
            useless_output = [v_t.assign(v) for v_t, v in zip(model.dense_layer2.variables, store_2)]
            useless_output = [v_t.assign(v) for v_t, v in zip(model.policy_layer.variables, store_3)]
                    
        model.update_target()
        
        
        if len(model.variables) != 24:
            sys.exit()
        
        score = 0
        for dd,idxx in enumerate(list(dones)):
            if idxx:
                timestep_total.append(timestep_tmp)
                score_total.append(score)
                d1.append(score)
                score = 0
            else:
                timestep_tmp += 1
                score += rewards[dd]
        

        score_tmp = np.mean(score_total[-np.sum(dones):])
        reward_history.append(score_tmp)
        reward_smooth.append(np.mean(d1))
        update_time += 1
        
        #### logger
        entropy_loss = model.entropy(states)
        policy_loss = model.loss_fn_policy(states, f_delta, entropy_par, K,C) + entropy_loss*entropy_par
        q_loss = model.loss_fn_q(states_sampled, target, actions_sampled, coef_q)
        tmp = tf.concat(
              [tf.reshape(g, [-1]) for g in grad_dir if g is not None], axis=0)
        pg_norm = tf.norm(tmp)
        tmp = tf.concat(
              [tf.reshape(g, [-1]) for g in grad_q if g is not None], axis=0)
        qg_norm = tf.norm(tmp)
        policy_loss_save.append(float(policy_loss))
        entropy_loss_save.append(float(entropy_loss))
        q_loss_save.append(float(q_loss))
        pg_norm_save.append(float(pg_norm))
        qg_norm_save.append(float(qg_norm))
         
        if IsPlot:
            time_pass = time.time() - time_tic
            time_tic = time.time()
            plt.subplot(321)
            plt.plot(np.array(policy_loss_save))
            plt.title("policy loss")                
            plt.subplot(322)
            plt.plot(np.array(entropy_loss_save))
            plt.title("entropy loss")    
            plt.subplot(323)
            plt.plot(np.array(pg_norm_save))
            plt.title("policy grad norm")    
            plt.subplot(324)
            plt.plot(np.array(qg_norm_save))
            plt.title("q network grad norm")
            plt.subplot(325)
            plt.plot(np.array(q_loss_save))
            plt.title("Q estimation loss")
            plt.subplot(326)
            plt.plot(range(sample_time),reward_history[:sample_time],'.',range(sample_time),reward_smooth[:sample_time])
            plt.show()
            print("EP: " + str(update_time) + " Current_Score: " + str(reward_history[-1])+'\n')
            print("EP: " + str(update_time) + " Smooth_Score: " + str(reward_smooth[-1]) + "         ",end="\n")
            print("EP: " + str(update_time) + " Used time: " + str(time_pass) + "         ",end="\n")