import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations_with_replacement
from itertools import product, permutations
import random

Sg = [1, 2, 3, 4, 5]
Sl = [(i, j) for i in range(1, 6) for j in range(1, 6)]
Ag = [-1, 0, 1]

def reward_function(sg, ag, sdelta):
    global_reward = 15.0 / sg - 1.0 * (ag == -1)
    delta_array = np.array([agent[0] for agent in sdelta])
    local_reward = np.mean(delta_array - 0.5 * (delta_array > sg))
    return global_reward + local_reward

def global_agent_transition(sg, ag):
    return np.clip(sg + ag, 1, 5)

def expected_local_agent_transition(sg, sdelta):
    new_sdelta = []
    sdelta_array = np.array(sdelta)
    a = np.where(sdelta_array[:, 1] <= sg, sdelta_array[:, 1], 
                 np.clip(sdelta_array[:, 1] + (sg - sdelta_array[:, 0]), 1, 5))
    b = np.full_like(a, 3)
    return list(zip(a, b))

def transition_function(sg, ag, sdelta):
    sdelta_array = np.array(sdelta)
    delta_random = np.random.randint(0, 2, size=sdelta_array.shape[0])
    a = np.where(sdelta_array[:, 1] <= sg, sdelta_array[:, 1], 
                 np.clip(sdelta_array[:, 1] + (sg - sdelta_array[:, 0]) * delta_random, 1, 5))
    b = np.random.randint(1, 6, size=sdelta_array.shape[0])
    return global_agent_transition(sg, ag), list(zip(a, b))

def generate_combinations_with_replacement(S_l, k):
    all_permutations = [tuple(sorted(p)) for p in product(S_l, repeat=k)]
    unique_permutations = set(all_permutations)
    return sorted(unique_permutations)

def make_Q_function(Sg, Sl, Ag, k):
    Q = {}
    Sdelta = generate_combinations_with_replacement(Sl, k)
    for sg in Sg:
        Q[sg] = {tuple(sdelta): {ag: 0 for ag in Ag} for sdelta in Sdelta}
    return Q, Sdelta

def Q_learning(Sg, Sl, Ag, k, gamma, bellman_iters):
    Qk, Sdelta = make_Q_function(Sg, Sl, Ag, k)
    for _ in tqdm(range(bellman_iters)):
        for sg in Sg:
            for sdelta in Sdelta:
                sdelta_tuple = tuple(sdelta)
                for ag in Ag:
                    exp_sg = global_agent_transition(sg, ag)
                    exp_sdelta = expected_local_agent_transition(sg, sdelta)
                    exp_sdelta_tuple = tuple(sorted(exp_sdelta))
                    avg_val = np.max(list(Qk[exp_sg][exp_sdelta_tuple].values()))
                    Qk[sg][sdelta_tuple][ag] = reward_function(sg, ag, sdelta) + gamma * avg_val
    return Qk

def Q_deploy(Sg, Sl, Ag, k, gamma, n, bellman_iters, T):
    Qk = Q_learning(Sg, Sl, Ag, k, gamma, bellman_iters)
    sn = [Sl[np.random.randint(0, len(Sl))] for _ in range(n)]
    sg = Sg[np.random.randint(0, len(Sg))]
    reward_samples = []
    for _ in range(10):
        total_reward = 0
        for _ in range(T):
            sdelta = random.sample(sn, k)
            sdelta_tuple = tuple(sorted(sdelta))
            ag = max(Qk[sg][sdelta_tuple], key=Qk[sg][sdelta_tuple].get)
            total_reward += gamma**T * reward_function(sg, ag, sdelta)
            sg, sn = transition_function(sg, ag, sn)
        reward_samples.append(total_reward)
    return reward_samples

expected_rewards = []
lowest_rewards = []
highest_rewards = []
for k in range(1, 8):
    reward_samples = Q_deploy(Sg, Sl, Ag, k=k, gamma=0.9, n=8, bellman_iters=10, T=200)
    expected_rewards.append(np.mean(reward_samples))
    lowest_rewards.append(np.min(reward_samples))
    highest_rewards.append(np.max(reward_samples))

expected_gap = np.log(expected_rewards[len(expected_rewards)-1] - expected_rewards)[:-1]
min_gap = np.log(lowest_rewards[len(lowest_rewards)-1] - lowest_rewards)[:-1]
max_gap = np.log(highest_rewards[len(highest_rewards)-1] - highest_rewards)[:-1]
lower_error = expected_gap - min_gap
upper_error = max_gap - expected_gap
print(expected_gap)
print(min_gap)
print(max_gap)
plt.figure()
plt.xlabel("k value")
plt.errorbar([i for i in range(1,6)], expected_gap, yerr=[lower_error, upper_error], fmt='o', capsize=5, capthick=2, ecolor='black')
plt.ylabel("Log reward optimality gap with error bars")
plt.savefig("small_scale_error_fig.png")