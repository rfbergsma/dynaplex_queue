#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers
@author: thomas
"""
import numpy as np
import torch
import random
import os
from shutil import copyfile


def graph_element_to_coordinate(location: int, size_x: int) -> dict:
    row = location // size_x
    return {'row': row,
            'column': location - (row * size_x)}


def distance(origin: int, destination: int, size_x: int) -> int:
    origin_coord = graph_element_to_coordinate(origin, size_x)
    destination_coord = graph_element_to_coordinate(destination, size_x)

    horizontal_dist = abs(origin_coord['row'] - destination_coord['row'])
    vertical_dist = abs(origin_coord['column'] - destination_coord['column'])

    return horizontal_dist + vertical_dist


# def state_dict_to_obs(state_dict, grid_size):
def state_dict_to_obs(state_dict, feats, grid_size, num_feats_per_node=8):
    # obs = np.zeros((grid_size * grid_size, num_feats_per_node), dtype=np.float32)
    mask = np.zeros(grid_size * grid_size)

    pickers = state_dict['picker_list']
    orders = state_dict['order_list'] if state_dict['order_list'] is not None else []
    current_picker = state_dict['current_picker']
    # current_distances = state_dict['current_distances']

    # # FILL OBSERVATION
    # n = 0
    #
    # # Current picker
    # obs[pickers[current_picker]["location"], 0] = 1
    #
    # # Distances from location of current agent
    # obs[:, 7] = current_distances
    #
    # unallocated_agents_locs = [picker["location"] for idx, picker in enumerate(pickers) if
    #                            (not picker["allocated"] and idx != current_picker)]
    # allocated_agents_locs = [picker["location"] for idx, picker in enumerate(pickers) if
    #                          (picker["allocated"] and idx != current_picker)]
    # allocated_agents_dests = np.array(
    #     [(picker["destination"], distance(picker["location"], picker["destination"], grid_size))
    #      for idx, picker in enumerate(pickers) if (picker["allocated"] and idx != current_picker)])
    #
    # n += 1
    #
    # # Unallocated agents
    # if len(unallocated_agents_locs) > 0:
    #     obs[:, n] = np.bincount(unallocated_agents_locs, minlength=np.size(obs[:, n]))
    # n += 1
    #
    # # Allocated agents
    # if len(allocated_agents_locs) > 0:
    #     obs[:, n] = np.bincount(allocated_agents_locs, minlength=np.size(obs[:, n]))
    # n += 1
    #
    # # Destinations
    # if len(allocated_agents_locs) > 0:
    #     obs[allocated_agents_dests[:, 0], n] = allocated_agents_dests[:, 1]
    # n += 1
    #
    # # Announced orders
    # announced_orders = np.array(
    #     [(order["location"], order["time_windows"][0]) for order in orders if order["state"] == 0])
    # if announced_orders.shape[0] > 0:
    #     obs[announced_orders[:, 0], n] = announced_orders[:, 1]
    # n += 1
    #
    # # Ongoing orders
    # ongoing_orders = np.array(
    #     [(order["location"], order["time_windows"][1]) for order in orders if order["state"] == 1])
    # if ongoing_orders.shape[0] > 0:
    #     obs[ongoing_orders[:, 0], n] = ongoing_orders[:, 1]
    # n += 1
    #
    # # Tardy orders
    # tardy_orders = np.array(
    #     [(order["location"], order["time_windows"][2] + 1) for order in orders if order["state"] == 2])
    # if tardy_orders.shape[0] > 0:
    #     obs[tardy_orders[:, 0], n] = tardy_orders[:, 1]
    # n += 1

    # FILL THE MASK
    mask[pickers[current_picker]["location"]] = 1

    # Only location of orders, since we only care about the destination for a pre-selected picker
    order_locs = [order["location"] for order in orders if order["assigned"] == -1]
    mask[order_locs] = 1

    return {'obs': torch.from_numpy(feats), 'mask': torch.tensor(mask, dtype=torch.bool).unsqueeze(0)}


def has_child_obs(state, obs, agent):

    found = False
    if agent in state.child_states:
        for idx, child_state in enumerate(state.child_states[agent]):
            # diff = np.linalg.norm(child_state.obs['obs'] - obs['obs'])
            diff = np.linalg.norm(child_state.obs['obs'] - obs['obs']) + \
                   np.linalg.norm(child_state.obs['mask'] ^ obs['mask'])
            if diff < 0.001:
                found = True
                return found, idx

    return found, 0


def stable_normalizer(x,temp):
    ''' Computes x[i]**temp/sum_i(x[i]**temp) '''
    # if np.max(x) == 0:
    #     print("qui")
    x = (x / np.max(x))**temp
    return np.abs(x/np.sum(x))

def argmax(x):
    ''' assumes a 1D vector x '''
    x = x.flatten()
    if np.any(np.isnan(x)):
        print('Warning: Cannot argmax when vector contains nans, results will be wrong')
    try:
        winners = np.argwhere(x == np.max(x)).flatten()   
        winner = random.choice(winners)
    except:
        winner = np.argmax(x)   # numerical instability ?
    return winner 


def store_safely(folder,name,to_store):
    ''' to prevent losing information due to interruption of process'''
    new_name = folder+name+'.npy'
    old_name = folder+name+'_old.npy'
    if os.path.exists(new_name):
        copyfile(new_name,old_name)
    np.save(new_name,to_store)
    if os.path.exists(old_name):            
        os.remove(old_name)

### Atari helpers ###
    
def get_base_env(env):
    ''' removes all wrappers '''
    while hasattr(env,'env'):
        env = env.env
    return env

def copy_atari_state(env):
    env = get_base_env(env)
    return env.clone_full_state()
#    return env.ale.cloneSystemState()

def restore_atari_state(env,snapshot):
    env = get_base_env(env)
    env.restore_full_state(snapshot)
#    env.ale.restoreSystemState(snapshot)

def is_atari_game(env):
    ''' Verify whether game uses the Arcade Learning Environment '''
    env = get_base_env(env)
    return hasattr(env,'ale')

### Database ##
    
class Database():
    ''' Database '''
    
    def __init__(self,max_size,batch_size):
        self.max_size = max_size        
        self.batch_size = batch_size
        self.clear()
        self.sample_array = None
        self.sample_index = 0
    
    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.size = 0
    
    def store(self,experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size +=1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def store_from_array(self,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)
        
    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0
                            
    def __iter__(self):
        return self

    def __next__(self):
        if (self.sample_index + self.batch_size > self.size) and (not self.sample_index == 0):
            self.reshuffle() # Reset for the next epoch
            raise(StopIteration)
          
        if (self.sample_index + 2*self.batch_size > self.size):
            indices = self.sample_array[self.sample_index:]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[self.sample_index:self.sample_index+self.batch_size]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size
        
        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add) 
        return tuple(arrays)
            
    next = __next__
    
### Visualization ##

def symmetric_remove(x,n):
    ''' removes n items from beginning and end '''
    odd = is_odd(n)
    half = int(n/2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x

def is_odd(number):
    ''' checks whether number is odd, returns boolean '''
    return bool(number & 1)

def smooth(y,window,mode):
    ''' smooth 1D vectory y '''    
    return np.convolve(y, np.ones(window)/window, mode=mode)