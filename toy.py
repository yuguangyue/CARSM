import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import random
import math
import scipy.stats as stats
import seaborn as sns; sns.set()
import time

def random_reward_function(a, magnitude, coef1, coef2, eps1, eps2, mid_point):
    part1 =  ((a > mid_point) & (a < 1)) * magnitude*(coef1*-((a-1) * (a-mid_point))) +\
    ((a > -1) & (a < mid_point)) * magnitude*(coef2*-((a+1)*(a-mid_point)))
    part2 = ((a > mid_point) & (a < 1) & (eps1 > 0)) * np.random.normal(loc = 0, scale = eps1, size = (len(a),)) +\
    ((a > -1) & (a < mid_point) & (eps2 > 0)) * np.random.normal(loc = 0, scale = eps2, size = (len(a),))
    return part1 + part2

def grad_randreward_func(a, magnitude, coef1, coef2, mid_point):
    return ((a > mid_point) & (a < 1)) * magnitude*(coef1*-(2*a-1-mid_point)) +\
    ((a > -1) & (a < mid_point)) * magnitude*(coef2*-(2*a+1-mid_point))
    
def polynomial_decay(lr_start, global_step, decay_steps, end_lr, power):
    global_step = min(global_step, decay_steps)
    decayed_learning_rate = (lr_start - end_lr) \
    * (1 - global_step / decay_steps) ** (power) \
    + end_lr
    return decayed_learning_rate

def softmax(phi):
    e_phi = np.exp(phi - np.max(phi))
    return e_phi / np.sum(e_phi)

def ent(p):
    return np.sum(-p * np.log(p))

def sample_dis_actions(actions):
    actions_ = []
    actions_idx = []
    for i, action in enumerate(actions):
        actions_.append(action_space[i]*np.ones((actions[i],1)))
        actions_idx.append(i * np.ones((actions[i],1)))
    actions_idx = np.vstack(actions_idx)
    actions_idx = actions_idx.astype(int)
    actions_ = np.reshape(np.vstack(actions_),(-1,))
    return actions_idx,actions_



batch_size = 100
magnitude = 1
mid_point = -0.8
theta_rec = []
sigma_rec = []

coef1 = 10/((1-mid_point)**2/4)
coef2 = 10.25/((1+mid_point)**2/4)
eps1 = 2
eps2 = 1

x = np.linspace(-1,1,101)
y = random_reward_function(x,magnitude,coef1,coef2,0,0,mid_point)
plt.plot(x,y)


## training for Gaussian policy
heat_map = np.zeros((10000,101)) # 8000 is number of iteration, and 101 is discretization precision.
dis_actions = np.linspace(-1,1,num=101)
fp = 0
tp = 0
lr = 0.01
for ite in range(100):
    samples = 0
    theta = mid_point
    sigma = 1
    ent_coef_start = 1
    total_samples = 1e6
    inner_iteration = 0
#    ent_coef = 1
    while (samples<1e6):
        ent_coef = polynomial_decay(ent_coef_start, samples, total_samples, 0, 2)
    
    ######################  reparametrization #####################################
        epsilon = np.random.normal(loc = 0, scale = 1, size = (batch_size,))
        actions = epsilon * sigma + theta        
        rewards = random_reward_function(actions,magnitude,coef1,coef2,eps1,eps2,mid_point)
        grad_theta = np.mean(grad_randreward_func(actions, magnitude, coef1, coef2,mid_point))
        grad_sigma = np.mean(grad_randreward_func(actions, magnitude, coef1, coef2,mid_point) * epsilon + ent_coef * 1 / sigma)
        
        heat_map[inner_iteration,:] += np.clip(stats.norm.pdf(dis_actions, theta, sigma), 0,3)
        # to prevent from overwhelming other actions
        
        theta += lr * grad_theta
        sigma += lr * grad_sigma
        sigma = np.max([1e-8, sigma])
    ###############################################################################
        
        samples += batch_size
        inner_iteration += 1
        theta_rec.append(theta)
        sigma_rec.append(sigma)
        
        if np.sqrt(grad_theta ** 2 + grad_sigma ** 2) < 1e-4:
            break

    print("this is: " + str(ite) + "   iteration         ",end="\n")
            
    if abs(theta - (mid_point+1)/2 ) < 1e-3:
        fp += 1
    elif abs(theta - (mid_point-1)/2 ) < 1e-3:
        tp += 1
plt.plot(np.array(theta_rec))
plt.plot(np.array(sigma_rec))

## training for discrete policy

tp,fp = 0,0
lr = 0.01
num_actions = 21
action_space = np.linspace(start=-1,stop=1,num=num_actions)
heat_map_dis = np.zeros((10000,num_actions))
t = time.time()
for ite in range(100):
    samples = 0
    phi = np.zeros((num_actions,))
    ent_coef_start = 1
    inner_iteration = 0
    while (samples<1e6):
        ent_coef = polynomial_decay(ent_coef_start, samples, total_samples, 0, 2)

        p = softmax(phi)
        actions = np.random.multinomial(100, p)
        actions_idx, actions_ = sample_dis_actions(actions)
        nb_classes = num_actions
        one_hot_targets = np.squeeze(np.eye(nb_classes)[actions_idx], axis = 1)
    
    ############################estimated gradient######################################
        grad_tmp = np.reshape(random_reward_function(actions_, magnitude, coef1, coef2, eps1, eps2,mid_point),(-1,1)) *\
        (one_hot_targets - p)

        grad = np.mean(grad_tmp, axis = 0)
        ## grad for entropy term
        coef_ent_grad = np.sum((np.log(p) + 1) * np.exp(phi)) / (np.sum(np.exp(phi)) ** 2)
        grad_ent = coef_ent_grad * np.exp(phi) - (np.log(p)+1) * p
        
        heat_map_dis[inner_iteration,:] += softmax(phi)
        phi += (grad + grad_ent * ent_coef) * lr        
    
        samples += batch_size
        inner_iteration+=1
        if np.sqrt(np.sum(grad**2)) < 1e-4:
            break
        
    p = softmax(phi)
    print("Use time: " + str(time.time() -t) + "       this is: " + str(ite) + "iteration         ",end="\n")
    t = time.time()
    if p[11] > 0.90:
        fp += 1
    elif p[1] > 0.90:
        tp += 1