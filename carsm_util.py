import tensorflow as tf
import numpy as np
import random
import gym
import tensorflow.contrib.eager as tfe
from carsm_util_2 import *
import multiprocessing
import sys
import baselines.common.tf_util as U
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from itertools import permutations, repeat
import itertools
from gym import utils
from baselines.common.cg import cg
def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init        

def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x



def runner_mujoco(env, nstep, model, gamma, state, nA, env_type, replay_buffer, K, C, high, low, random_choose,one_hot):
    rewards, actions, states, dones = [],[],[],[]
    noised_states = []
    phi_sequence, pi_sequence, actions_idx=[],[],[]
    n_step = 0
    last_done_idx = 0
    gap = (high - low) / (C-1)       
    while n_step < nstep:

        phi = model.policy(tf.convert_to_tensor(state))[0]
        phi = tf.reshape(phi, (K,C))
        pi = np.random.dirichlet(np.ones(C), size = (K))
        action_true_idx = np.argmin(np.log(pi) -phi, axis = 1)
        
        if random_choose:
            action = np.float32(low + np.random.uniform(0,gap) + action_true_idx * gap)
        else:
            action = np.int32(low + action_true_idx * (high - low) / (C-1))#for classical lunarlander
            #action = np.float32(low + action_true_idx * (high - low) / (C-1))
        
        next_state,reward,done,_ = env.step(action)
#        next_state = np.float32(next_state)
        reward = np.float32(reward)        
        if one_hot:
            replay_buffer.add(state, action_true_idx, reward, next_state, np.float32(done))
        else:
            replay_buffer.add(state, action, reward, next_state, np.float32(done))
            
        ### add action_true_idx on 07/21
    
#        phi_sequence.append(phi)
        phi_sequence.append(np.array(phi)) ## changed at 05/10
        pi_sequence.append(pi)
        actions.append(action)
        states.append(state)
        rewards.append(reward)
        dones.append(done)
        actions_idx.append(np.float32(action_true_idx))
        

        state = next_state
        n_step += 1
        if done:
            last_done_idx = n_step
            last_state = state
#            break
    last_state = state
    last_done = done
    ## if we truncate the last state to done, then I don't really necessary to use last_state
#    rewards = rewards[:last_done_idx]
#    actions = actions[:last_done_idx]
#    states = states[:last_done_idx]
#    dones = dones[:last_done_idx]
#    phi_sequence = phi_sequence[:last_done_idx]
#    pi_sequence = pi_sequence[:last_done_idx]
#    actions_idx = actions_idx[:last_done_idx]
    

    return rewards, actions, states, dones, phi_sequence, pi_sequence, last_state, actions_idx, last_done


##### functions copy from baselines
def build_env(num_env,alg,seed,env_type,env_id,reward_scale,gamestate=None):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = num_env or ncpu

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=gamestate, reward_scale=reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, num_env or 1, seed, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env    





class ARSM_AGENT_mujoco(tf.keras.Model):
    def __init__(self, nA, network, tau, Grad_Clip_Norm, K, C):
        super(ARSM_AGENT_mujoco, self).__init__()
        """ Define here the layers used during the forward-pass 
            of the neural network.
        """
        self.network = network
        self.tau = tau
        self.K = K
        self.C = C
        self.Grad_Clip_Norm = Grad_Clip_Norm
        if self.network == 'Linear':        
            # Hidden layer.
            self.dense_layer1 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal())
            self.dense_layer2 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer3 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer4 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            # Output layer. No activation.
            self.policy_layer = tf.layers.Dense(nA, activation=None, kernel_initializer=tf.keras.initializers.he_normal())
            self.q_layer = tf.layers.Dense(1, activation=None,kernel_initializer=tf.keras.initializers.he_normal())
           
            ### target network
            self.dense_layer_c1 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal())
            self.dense_layer_c2 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer_c3 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer_c4 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            # Output layer. No activation.
            self.policy_layer_c = tf.layers.Dense(nA, activation=None, kernel_initializer=tf.keras.initializers.he_normal())
            self.q_layer_c = tf.layers.Dense(1, activation=None,kernel_initializer=tf.keras.initializers.he_normal())            
            
            
        elif self.network == 'Convolution':
            self.cnn1 = tf.layers.Conv2D(32, 8, 4, kernel_initializer=ortho_init(np.sqrt(2)),\
                                  activation = tf.nn.tanh, padding = 'valid')
            self.cnn2 = tf.layers.Conv2D(64, 4, 2,kernel_initializer=ortho_init(np.sqrt(2)),\
                                         activation = tf.nn.tanh, padding = 'valid')
            self.cnn3 = tf.layers.Conv2D(64, 3, 1,\
                                  kernel_initializer=ortho_init(np.sqrt(2)), activation = tf.nn.relu,\
                                  padding = 'valid')
            self.dense_layer = tf.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2))) 
            # Output layer. No activation.
            self.policy_layer = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))
            self.q_layer = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))
            
            ### target network            
            self.cnn_c1 = tf.layers.Conv2D(32, 8, 4, kernel_initializer=ortho_init(np.sqrt(2)),\
                                  activation = tf.nn.relu, padding = 'valid')
            self.cnn_c2 = tf.layers.Conv2D(64, 4, 2,kernel_initializer=ortho_init(np.sqrt(2)),\
                                         activation = tf.nn.relu, padding = 'valid')
            self.cnn_c3 = tf.layers.Conv2D(64, 3, 1,\
                                  kernel_initializer=ortho_init(np.sqrt(2)), activation = tf.nn.relu,\
                                  padding = 'valid')
            self.dense_layer_c = tf.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2))) 
            # Output layer. No activation.
            self.policy_layer_c = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))
            self.q_layer_c = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))            
    def duplicate(self):
        if self.network == 'Linear':        
            # copy the parameters
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c1.variables, self.dense_layer1.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c2.variables, self.dense_layer2.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c3.variables, self.dense_layer3.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c4.variables, self.dense_layer4.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]                            
        elif self.network == 'Convolution':
            [v_t.assign(v) for v_t, v in zip(self.cnn_c1.variables, self.cnn1.variables)]
            [v_t.assign(v) for v_t, v in zip(self.cnn_c2.variables, self.cnn2.variables)]
            [v_t.assign(v) for v_t, v in zip(self.cnn_c3.variables, self.cnn3.variables)]
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c.variables, self.dense_layer.variables)]
            [v_t.assign(v) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]
            [v_t.assign(v) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]
            
        
    def  dup_q_estimation(self,state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer_c3(state)
            h2 = self.dense_layer_c4(h1)
            q_values = self.q_layer_c(h2)        

            return q_values
        elif self.network == 'Convolution':
            h1 = self.cnn_c1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn_c2(h1)
            h3 = self.cnn_c3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer_c(h3)
            q_values = self.q_layer_c(h4)
        
            return q_values        
        
    def dup_policy(self, state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer_c1(state)
            h2 = self.dense_layer_c2(h1)
            logits = self.policy_layer_c(h2)

            return logits
        elif self.network == 'Convolution':
            h1 = self.cnn_c1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn_c2(h1)
            h3 = self.cnn_c3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer_c(h3)
            logits = self.policy_layer_c(h4)
        
            return logits
    
    def update_target(self):
        if self.network == 'Linear':
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c1.variables, self.dense_layer1.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c2.variables, self.dense_layer2.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c3.variables, self.dense_layer3.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c4.variables, self.dense_layer4.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]      
        elif self.network == 'Convolution':                      
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.cnn_c1.variables, self.cnn1.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.cnn_c2.variables, self.cnn2.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.cnn_c3.variables, self.cnn3.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c.variables, self.dense_layer.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]   

    
    def policy(self, state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer1(state)
            h2 = self.dense_layer2(h1)
            logits = self.policy_layer(h2)

            return logits
        elif self.network == 'Convolution':
            h1 = self.cnn1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn2(h1)
            h3 = self.cnn3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer(h3)
            logits = self.policy_layer(h4)
        
            return logits
    
    def q_estimation(self, state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer3(state)
            h2 = self.dense_layer4(h1)
            q_values = self.q_layer(h2)        

            return q_values
        elif self.network == 'Convolution':
            h1 = self.cnn1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn2(h1)
            h3 = self.cnn3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer(h3)
            q_values = self.q_layer(h4)
        
            return q_values        

    
    def loss_fn_q(self, states, q_target, actions, coef_q):
        """ Defines the loss function used during 
            training.         
        """
        q_input = tf.concat([states,actions], axis =1)
        q_values= self.q_estimation(q_input)
        q_values = tf.reshape(q_values, [-1])
        loss = coef_q * tf.reduce_mean(tf.square(q_values - q_target))
        return loss
    
    def grads_fn_q(self, states, q_target, actions,coef_q):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn_q(states, q_target, actions, coef_q)
        return tape.gradient(loss, self.variables)
    
    def entropy(self, states):
        logits = self.policy(states)
        K = self.K
        C = self.C
        logits = tf.reshape(logits, (-1,K,C))
        ent = np.float32(0)
        for kk in range(K):
            logits_tmp = logits[:,kk,:]
            a0 = logits_tmp - tf.reduce_max(logits_tmp, 1, keep_dims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
            p0 = ea0 / z0
            ent += tf.reduce_sum(p0 * (tf.log(z0) - a0))
        #return ent/K 
        return ent
    
    def entropy_pseudo_states(self, pseudo_states_idx_save, pseudo_actions_idx_save,states,actions_idx): 
        logits = self.policy(states)
        K = self.K
        C = self.C
        logits = tf.reshape(logits, (-1,K,C))    
        logits = tf.gather_nd(logits, np.reshape(pseudo_states_idx_save,(-1,1))) # only select states has pseudo actions
        logits1 = logits * tf.one_hot(pseudo_actions_idx_save,C) # pseudo action
        tmp = actions_idx[pseudo_states_idx_save]
        logits2 = logits * tf.one_hot(tmp.astype(int),C) # true action
        logits = logits1 + logits2        
        ent = np.float32(0)
        for kk in range(K):
            logits_tmp = logits[:,kk,:]
            b0 = tf.where( tf.equal(0, logits_tmp), -1000 * tf.ones_like( logits_tmp ), logits_tmp)
            a0 = b0 - tf.reduce_max(b0, 1, keep_dims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
            p0 = ea0 / z0
            ent += tf.reduce_sum(p0 * (tf.log(z0) - a0))
        return ent
    
    def loss_fn_policy(self, states, f_delta, ent_par, K, C):
        logit = self.policy(states)
        logit = tf.reshape(logit, (-1, K, C))
        logit = tf.dtypes.cast(logit,tf.float32)
#        logit = tf.reshape(logit, (-1, C, K))
        ent = self.entropy(states)
        ent = tf.dtypes.cast(ent,tf.float32)

        
        #return tf.reduce_sum(tf.multiply(logit, f_delta)) - ent_par*ent
        return tf.reduce_mean(tf.multiply(logit, f_delta)) - (ent_par/(K*len(states)))*ent
 
    
    def loss_fn_policy_pseudo_states(self, pseudo_states_idx_save, pseudo_actions_idx_save,states,actions_idx,f_delta,ent_par,K,C):
        logit = self.policy(states)
        logit = tf.reshape(logit, (-1, K, C))
#        logit = tf.reshape(logit, (-1, C, K))
        ent = self.entropy_pseudo_states(pseudo_states_idx_save, pseudo_actions_idx_save,states,actions_idx)
       
        return tf.reduce_mean(tf.multiply(logit, f_delta)) - (ent_par/(K*len(states)))*ent        
        

    def grads_fn_policy(self, states, f_delta, ent_par, K, C):

        with tf.GradientTape() as tape:
            loss = self.loss_fn_policy(states, f_delta, ent_par, K, C)
        return tape.gradient(loss, self.variables)
        
    def grads_fn_policy_pseudo_states(self, pseudo_states_idx_save, pseudo_actions_idx_save,states,actions_idx,f_delta,ent_par,K,C):

        with tf.GradientTape() as tape:
            loss = self.loss_fn_policy_pseudo_states(pseudo_states_idx_save, pseudo_actions_idx_save,states,actions_idx,f_delta,ent_par,K,C)
        return tape.gradient(loss, self.variables)
    
    
    def update_policy(self, states, optimizer, f_delta, ent_par, K, C):
        
        grad_policy = self.grads_fn_policy(states, f_delta, ent_par, K, C)
        optimizer.apply_gradients(zip(grad_policy, self.variables))
        return grad_policy
    
    def update_policy_pseudo_states(self,optimizer,pseudo_states_idx_save, pseudo_actions_idx_save,states,actions_idx,f_delta,ent_par,K,C):
        
        grad_policy = self.grads_fn_policy(states, f_delta, ent_par, K, C)
        optimizer.apply_gradients(zip(grad_policy, self.variables))
        return grad_policy    
        
    def update_q(self, states, optimizer, q_target, Grad_Clip_NN, actions, coef_q):
        
        grad_q = self.grads_fn_q(states, q_target, actions, coef_q)
        if Grad_Clip_NN:
            grad_q = tf.clip_by_global_norm(grad_q, self.Grad_Clip_Norm)[0]
        optimizer.apply_gradients(zip(grad_q, self.variables))
        return grad_q
    
    def loss_q_prioritize(self, states, q_target, actions, coef_q, weights):
        q_values= self.q_estimation(states)
        q_values = tf.reduce_sum(tf.multiply(q_values,actions),axis = 1)
        loss = coef_q * tf.reduce_mean(weights * U.huber_loss((q_values - q_target)))
        return loss        
    
    def grads_fn_q_prioritize(self, states, q_target, actions,coef_q,weights):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tf.GradientTape() as tape:
            loss = self.loss_q_prioritize(states, q_target, actions, coef_q, weights)
        return tape.gradient(loss, self.variables)    

    def update_q_prioritize(self, states, optimizer, q_target, Grad_Clip_NN, actions, coef_q, weights):
        grad_q = self.grads_fn_q_prioritize(states, q_target, actions,coef_q,weights)
        if Grad_Clip_NN:
            grad_q = tf.clip_by_global_norm(grad_q, self.Grad_Clip_Norm)[0]
        optimizer.apply_gradients(zip(grad_q, self.variables))

        
        
def expected_sarsa(model,state,K,C,low,high,dup,random_choose,num,noise_injection=False): ## in mujoco, use MC to estimate the expectation.         
    if noise_injection:
        white_noise = np.random.normal(loc=0,scale=0,size=(1,17))
        if dup:
            logits = np.array(tf.nn.softmax(tf.reshape(model.dup_policy(state,white_noise), (K,C)), axis = 1))
        else:
            logits = np.array(tf.nn.softmax(tf.reshape(model.policy(state,white_noise), (K,C)), axis = 1))
    else:
        if dup:
            logits = np.array(tf.nn.softmax(tf.reshape(model.dup_policy(state), (K,C)), axis = 1))
        else:
            logits = np.array(tf.nn.softmax(tf.reshape(model.policy(state), (K,C)), axis = 1))
    low = np.reshape(low, (-1,1))
    high = np.reshape(high, (-1,1))
    ## sample process
    action_samples=[np.random.choice(range(C),size=num, p=logits[i]) for i in range(K)] ### check
    action_samples = np.float32(np.vstack(action_samples))
    gap = (high - low) / (C-1)
    if random_choose:
        actions = action_samples * gap + low +  np.float32(np.random.uniform(0,gap[0],size=(action_samples.shape)))
    else:
        actions = action_samples * gap + low
    actions = actions.T

    state_tile = np.tile(state, (num,1))
    input_q = np.concatenate((state_tile,actions), axis = 1)
    if dup:
        q_value = model.dup_q_estimation(input_q)
    else:
        q_value = model.q_estimation(input_q)
        
    return np.float32(np.array(tf.reduce_mean(q_value)))

def expected_sarsa_one_hot(model,state,K,C,low,high,dup,random_choose,action_one_hot,num): ## in mujoco, use MC to estimate the expectation.         
    if dup:
        logits = np.array(tf.nn.softmax(tf.reshape(model.dup_policy(state), (K,C)), axis = 1))
    else:
        logits = np.array(tf.nn.softmax(tf.reshape(model.policy(state), (K,C)), axis = 1))
    low = np.reshape(low, (-1,1))
    high = np.reshape(high, (-1,1))
    ## sample process
    action_samples=[np.random.choice(range(C),size=num, p=logits[i]) for i in range(K)] ### check
    action_samples = np.float32(np.vstack(action_samples))
    gap = (high - low) / (C-1)
    if random_choose:
        actions = action_samples * gap + low +  np.float32(np.random.uniform(0,gap[0],size=(action_samples.shape)))
    else:
        actions = action_samples * gap + low
    actions = actions.T
    actions = action_one_hot(actions,K,C)
    actions = np.float32(actions)
    state_tile = np.tile(state, (num,1))
    input_q = np.concatenate((state_tile,actions), axis = 1)
    if dup:
        q_value = model.dup_q_estimation(input_q)
    else:
        q_value = model.q_estimation(input_q)
        
    return np.float32(np.array(tf.reduce_mean(q_value)))


def expected_sarsa_exact(model,state,K,C,low,high,dup,num): # only work for K = 2.. or 1 if commented
    # calculate pairwise expectation
    if dup:
        logits = np.array(tf.nn.softmax(tf.reshape(model.dup_policy(state), (K,C)), axis = 1))
    else:
        logits = np.array(tf.nn.softmax(tf.reshape(model.policy(state), (K,C)), axis = 1))
    ## for probability
    a = list(logits[0])
#    b = list(logits[1])
#    inputdata = [a,b]
    inputdata = [a]
    tmp=list(itertools.product(*inputdata))
    prob_unique = np.vstack(tmp)
    prob_tmp = np.reshape(np.prod(prob_unique, axis = 1),(-1,1))
    ## for actions
    a = list(range(C))
    inputdata = [a]
    tmp=list(itertools.product(*inputdata))
    action_unique = np.vstack(tmp)

    gap = (high - low) / (C-1)
    action_unique = np.float32(action_unique * gap + low)
    state_tmp_tile = np.tile(state, (len(action_unique),1))                
    input_tmp = np.concatenate((state_tmp_tile, action_unique), axis = 1)
    if dup:
        q_value = model.dup_q_estimation(input_tmp)
    else:
        q_value = model.q_estimation(input_tmp)
    expected_q = q_value * prob_tmp
    return tf.reduce_mean(expected_q)
    
def parallel_calculate_Q_part1_eliminateOneDim(action_idx_true,pi,phi,state,true_Q_target,C,K,random_choose,gap,action_low):
    pseudo_idx=[pseudo_action_swap_matrix_multi(pi[i],phi[i]) for i in range(K)] ## 05/10a
    dim_variation = -np.ones(K)
    for i in range(K):
        if np.sum(pseudo_idx[i] != action_idx_true[i]) == 0:
            dim_variation[i] = 0
        else:
            dim_variation[i] = 1
    pseudo_idx = np.vstack(pseudo_idx)
    pseudo_idx= [[pseudo_idx[:,i]] for i in range(C*C)]
    pseudo_idx = np.vstack(pseudo_idx)
    temp= np.unique(pseudo_idx, axis=0)
    pseudo_idx_ = (np.sum(temp==action_idx_true, axis=1)!=K)
    temp = temp[pseudo_idx_] ## unique pseudo actions
    return temp,pseudo_idx,dim_variation
    
def parallel_calculate_Q_part2_eliminateOneDim(Q_tmp,temp,pseudo_idx,true_Q_target,pi,dim_variation,C,K):
    ft = np.full(C*C,true_Q_target)
    if len(temp)> 0:
        for idx,ii in enumerate(temp):
            ft[find_location(pseudo_idx, ii, K)]= np.array(Q_tmp[idx])
        ft = np.reshape(ft,(C,C))
        meanft = np.mean(ft,axis=0)
        f = np.zeros([1,K,C])
        for kk in range(K):
            if dim_variation[kk]:
                f[0,kk,:] = np.matmul(ft-meanft,1.0/C-pi[kk])
            else:
                f[0,kk,:] = 0
        return f
    else:
        return np.zeros([1,K,C])    
    
def check_nan(a_list):
    rec = 0
    for a in a_list:
        rec += sum(np.isnan(a))
    return rec

    