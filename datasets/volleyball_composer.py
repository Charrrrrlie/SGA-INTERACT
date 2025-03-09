import os
import pdb
import sys
from pathlib import Path 
import pickle
import numpy as np
import random
import copy
import json
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from datasets.utils import personwise_normalization, oks_one_keypoint_compute
from datasets.utils import horizontal_flip_augment_joint, horizontal_flip_ball_trajectory
from datasets.utils import horizontal_move_augment_joint, vertical_move_augment_joint, agent_dropout_augment_joint
from datasets.constants import VOLLEYBALL_ACTION_DICT

class Volleyball_COMPOSER(Dataset):
    def __init__(self, args, logger, split='train'):
        self.args = args
        self.split = split
        self.logger = logger
        
        if args.olympic_split:
            self.dataset_splits = {
                'train': [1, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                          41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                'test': [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                         19, 20, 21, 22, 23, 24, 25, 26, 27]
            }
        else:
            self.dataset_splits = {
                'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39,
                          40, 41, 42, 48, 50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26,
                          27, 28, 30, 33, 46, 49, 51],
                'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
            }

        # two types of actions _ and -
        self.idx2class = VOLLEYBALL_ACTION_DICT
        self.class2idx = dict()
        if self.logger is not None:
            self.logger.info(f'Initializing Volleyball dataset with {split} split')
            self.logger.info('class index:')
        for k, v_list in self.idx2class.items():
            for v in v_list:
                self.class2idx[v] = k
                if self.logger is not None:
                    self.logger.info('{}: {}'.format(v, k))

        self.person_actions_all = pickle.load(
                open(os.path.join(self.args.path, 'tracks_normalized_with_person_action_label.pkl'), "rb"))
        # ACTIONS = ['NA', 'blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']
        # { 'NA': 0,
        # 'blocking': 1, 
        # 'digging': 2, 
        #  'falling': 3, 
        #  'jumping': 4,
        #  'moving':5 , 
        #  'setting': 6, 
        #  'spiking': 7, 
        #  'standing': 8,
        #  'waiting': 9}
        
        self.annotations = []
        self.annotations_each_person = []
        self.clip_joints_paths = []
        self.clips = []
        if args.ball_trajectory_use:
            self.clip_ball_paths = []
        self.prepare(args.path)
            
        if self.args.horizontal_flip_augment and self.split == 'train':
            self.classidx_horizontal_flip_augment = {
                0: 1,
                1: 0,
                2: 3,
                3: 2,
                4: 5,
                5: 4,
                6: 7,
                7: 6
            }
            if self.args.horizontal_flip_augment_purturb:
                self.horizontal_flip_augment_joint_randomness = dict()
                
        if self.args.horizontal_move_augment and self.split == 'train':
            self.horizontal_move_augment_joint_randomness = dict()
                
        if self.args.vertical_move_augment and self.split == 'train':
            self.vertical_move_augment_joint_randomness = dict()
            
        if self.args.agent_dropout_augment:
            self.agent_dropout_augment_randomness = dict()
            
        self.collect_standardization_stats()
        
        
        self.tdata = pickle.load(
            open(os.path.join(self.args.path, 'tracks_normalized.pkl'), "rb"))

    def prepare(self, dataset_dir):
        """
        Prepare the following lists based on the dataset_dir, self.split
            - self.annotations 
            - self.annotations_each_person 
            - self.clip_joints_paths
            - self.clips
            (the following if needed)
            - self.clip_ball_paths
            - self.horizontal_flip_mask
            - self.horizontal_mask
            - self.vertical_mask
            - self.agent_dropout_mask
        """  
        annotations_thisdatasetdir = defaultdict()
        clip_joints_paths = []

        for annot_file in glob.glob(os.path.join(dataset_dir, 'videos/*/annotations.txt')):
            video = annot_file.split('/')[-2]
            with open(annot_file, 'r') as f:
                lines = f.readlines()
            for l in lines:
                clip, label = l.split()[0].split('.jpg')[0], l.split()[1]
                annotations_thisdatasetdir[(video, clip)] = self.class2idx[label]  

        for video in self.dataset_splits[self.split]:
            clip_joints_paths.extend(glob.glob(os.path.join(dataset_dir, 'joints', str(video), '*.pickle')))

        count = 0
        for path in clip_joints_paths:
            video, clip = path.split('/')[-2], path.split('/')[-1].split('.pickle')[0]
            self.clips.append((video, clip))
            self.annotations.append(annotations_thisdatasetdir[(video, clip)])
            self.annotations_each_person.append(self.person_actions_all[(int(video), int(clip))])
            if self.args.ball_trajectory_use:
                self.clip_ball_paths.append(os.path.join(dataset_dir, 'volleyball_ball_annotation', video, clip + '.txt'))
            count += 1
        # print('total number of clips is {}'.format(count))

        self.clip_joints_paths += clip_joints_paths
      
        assert len(self.annotations) == len(self.clip_joints_paths)
        assert len(self.annotations) == len(self.annotations_each_person)
        assert len(self.clip_joints_paths) == len(self.clips)
        if self.args.ball_trajectory_use:
            assert len(self.clip_joints_paths) == len(self.clip_ball_paths)
        
        true_data_size = len(self.annotations)
        true_annotations = copy.deepcopy(self.annotations)
        true_annotations_each_person = copy.deepcopy(self.annotations_each_person)
        true_clip_joints_paths = copy.deepcopy(self.clip_joints_paths)
        true_clips = copy.deepcopy(self.clips)
        if self.args.ball_trajectory_use:
            true_clip_ball_paths = copy.deepcopy(self.clip_ball_paths)
        
        # if horizontal flip augmentation and is training
        if self.args.horizontal_flip_augment and self.split == 'train':
            self.horizontal_flip_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
                
        # if horizontal move augmentation and is training
        if self.args.horizontal_move_augment and self.split == 'train':
            self.horizontal_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
      
        # if vertical move augmentation and is training
        if self.args.vertical_move_augment and self.split == 'train':
            self.vertical_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
                
        # if random agent dropout augmentation and is training
        if self.args.agent_dropout_augment and self.split == 'train':
            self.agent_dropout_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths        
    
    def __len__(self):
        return len(self.clip_joints_paths)

    def collect_standardization_stats(self):
        # get joint x/y mean, std over train set
        if self.split == 'train':
            if self.args.recollect_stats_train or (
                not os.path.exists(os.path.join(self.args.path, 'prepare_stats', 'stats_train.pickle'))):
                if self.logger is not None:
                    self.logger.info('Collecting statistics for standardization...')
                joint_xcoords = []
                joint_ycoords = []
                joint_dxcoords = []
                joint_dycoords = [] 
                if self.args.ball_trajectory_use:
                        ball_xcoords = []
                        ball_ycoords = []
                        ball_dxcoords = []
                        ball_dycoords = [] 

                for index in range(self.__len__()):   # including augmented data!
                    with open(self.clip_joints_paths[index], 'rb') as f:
                        joint_raw = pickle.load(f)
                        
                    frames = sorted(joint_raw.keys())[self.args.frame_start_idx:self.args.frame_end_idx+1:self.args.frame_sampling]

                    if self.args.ball_trajectory_use:
                        ball_trajectory_data = self.read_ball_trajectory(self.clip_ball_paths[index])
                        ball_trajectory_data = ball_trajectory_data[self.args.frame_start_idx+10:self.args.frame_end_idx+1+10:self.args.frame_sampling]
                        # ball trajectory annotation has 41 frames annotated but joint/track file only has 20 frames.
                        assert len(ball_trajectory_data) == len(frames)
                        # (T, 2)
                       
                    # if horizontal flip augmentation and is training
                    if self.args.horizontal_flip_augment:
                        if index < len(self.horizontal_flip_mask):
                            if self.horizontal_flip_mask[index]:
                                if self.args.horizontal_flip_augment_purturb:
                                    self.horizontal_flip_augment_joint_randomness[index] = defaultdict()
                                    joint_raw = horizontal_flip_augment_joint(
                                        joint_raw, frames, self.horizontal_flip_augment_joint_randomness, self.args.image_w,
                                        add_purturbation=True, randomness_set=False, index=index)
                                else:
                                    joint_raw = horizontal_flip_augment_joint(joint_raw, frames, self.horizontal_flip_augment_joint_randomness, self.args.image_w)

                                if self.args.ball_trajectory_use:
                                    ball_trajectory_data = horizontal_flip_ball_trajectory(ball_trajectory_data, self.args.image_w)
                                    
                    
                    # if horizontal move augmentation and is training
                    if self.args.horizontal_move_augment:
                        if index < len(self.horizontal_mask):
                            if self.horizontal_mask[index]:
                                if self.args.ball_trajectory_use:
                                    if self.args.horizontal_move_augment_purturb:
                                        self.horizontal_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw, ball_trajectory_data = horizontal_move_augment_joint(
                                            joint_raw, frames, self.horizontal_move_augment_joint_randomness,
                                            add_purturbation=True, randomness_set=False, index=index, ball_trajectory=ball_trajectory_data)
                                    else:
                                        joint_raw, ball_trajectory_data = horizontal_move_augment_joint(joint_raw, frames, self.horizontal_move_augment_joint_randomness,
                                                                                                        ball_trajectory=ball_trajectory_data)
                                else:
                                    if self.args.horizontal_move_augment_purturb:
                                        self.horizontal_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw = horizontal_move_augment_joint(
                                            joint_raw, frames,  self.horizontal_move_augment_joint_randomness,
                                            add_purturbation=True, randomness_set=False, index=index)
                                    else:
                                        joint_raw = horizontal_move_augment_joint(joint_raw, frames, self.horizontal_move_augment_joint_randomness)
                            
                    # if vertical move augmentation and is training
                    if self.args.vertical_move_augment:
                        if index < len(self.vertical_mask):
                            if self.vertical_mask[index]:
                                if self.args.ball_trajectory_use:
                                    if self.args.vertical_move_augment_purturb:
                                        self.vertical_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw, ball_trajectory_data = vertical_move_augment_joint(
                                            joint_raw, frames, self.vertical_move_augment_joint_randomness,
                                            add_purturbation=True, randomness_set=False, index=index, ball_trajectory=ball_trajectory_data)
                                    else:
                                        joint_raw, ball_trajectory_data = vertical_move_augment_joint(joint_raw, frames, self.vertical_move_augment_joint_randomness,
                                                                                                      ball_trajectory=ball_trajectory_data)
                                else:
                                    if self.args.vertical_move_augment_purturb:
                                        self.vertical_move_augment_joint_randomness[index] = defaultdict()
                                        joint_raw = vertical_move_augment_joint(
                                            joint_raw, frames, self.vertical_move_augment_joint_randomness, 
                                            add_purturbation=True, randomness_set=False, index=index)
                                    else:
                                        joint_raw = vertical_move_augment_joint(joint_raw, frames, self.vertical_move_augment_joint_randomness,)
                                    
                    # To compute statistics, no need to consider the random agent dropout augmentation,
                    # but we can set the randomness here.
                    # if random agent dropout augmentation and is training
                    if self.args.agent_dropout_augment:
                        if index < len(self.agent_dropout_mask):
                            if self.agent_dropout_mask[index]:
                                chosen_frame = random.choice(frames)
                                chosen_person = random.choice(range(self.args.N))
                                self.agent_dropout_augment_randomness[index] = (chosen_frame, chosen_person)
            
                    
                    joint_raw = self.joints_sanity_fix(joint_raw, frames)
                    if self.args.ball_trajectory_use:
                        ball_trajectory_data = self.ball_trajectory_sanity_fix(ball_trajectory_data)
                    

                    for tidx, frame in enumerate(frames):
                        joint_xcoords.extend(joint_raw[frame][:,:,0].flatten().tolist())
                        joint_ycoords.extend(joint_raw[frame][:,:,1].flatten().tolist())

                        if tidx != 0:
                            pre_frame = frames[tidx-1]
                            joint_dxcoords.extend((joint_raw[frame][:,:,0]-joint_raw[pre_frame][:,:,0]).flatten().tolist())
                            joint_dycoords.extend((joint_raw[frame][:,:,1]-joint_raw[pre_frame][:,:,1]).flatten().tolist())
                        else:
                            joint_dxcoords.extend((np.zeros((self.args.N, self.args.J))).flatten().tolist())
                            joint_dycoords.extend((np.zeros((self.args.N, self.args.J))).flatten().tolist())
                            
                    if self.args.ball_trajectory_use:
                        ball_xcoords.extend(list(ball_trajectory_data[:, 0]))
                        ball_ycoords.extend(list(ball_trajectory_data[:, 1]))
                        
                        for t in range(len(ball_trajectory_data)):
                            if t == 0:
                                ball_dxcoords.append(0)
                                ball_dycoords.append(0)
                            else:
                                ball_dxcoords.append(ball_trajectory_data[t, 0] - ball_trajectory_data[t-1, 0])
                                ball_dycoords.append(ball_trajectory_data[t, 1] - ball_trajectory_data[t-1, 1])
                             

                # -- collect mean std
                if self.args.ball_trajectory_use:
                    joint_xcoords_mean, joint_xcoords_std = np.mean(joint_xcoords), np.std(joint_xcoords)
                    joint_ycoords_mean, joint_ycoords_std = np.mean(joint_ycoords), np.std(joint_ycoords)
                    joint_dxcoords_mean, joint_dxcoords_std = np.mean(joint_dxcoords), np.std(joint_dxcoords)
                    joint_dycoords_mean, joint_dycoords_std = np.mean(joint_dycoords), np.std(joint_dycoords)
                    
                    ball_xcoords_mean, ball_xcoords_std = np.mean(ball_xcoords), np.std(ball_xcoords)
                    ball_ycoords_mean, ball_ycoords_std = np.mean(ball_ycoords), np.std(ball_ycoords)
                    ball_dxcoords_mean, ball_dxcoords_std = np.mean(ball_dxcoords), np.std(ball_dxcoords)
                    ball_dycoords_mean, ball_dycoords_std = np.mean(ball_dycoords), np.std(ball_dycoords) 


                    self.stats = {
                        'joint_xcoords_mean': joint_xcoords_mean, 'joint_xcoords_std': joint_xcoords_std,
                        'joint_ycoords_mean': joint_ycoords_mean, 'joint_ycoords_std': joint_ycoords_std,
                        'joint_dxcoords_mean': joint_dxcoords_mean, 'joint_dxcoords_std': joint_dxcoords_std,
                        'joint_dycoords_mean': joint_dycoords_mean, 'joint_dycoords_std': joint_dycoords_std,
                        'ball_xcoords_mean': ball_xcoords_mean, 'ball_xcoords_std': ball_xcoords_std,
                        'ball_ycoords_mean': ball_ycoords_mean, 'ball_ycoords_std': ball_ycoords_std,
                        'ball_dxcoords_mean': ball_dxcoords_mean, 'ball_dxcoords_std': ball_dxcoords_std,
                        'ball_dycoords_mean': ball_dycoords_mean, 'ball_dycoords_std': ball_dycoords_std
                    }

                else:
                    joint_xcoords_mean, joint_xcoords_std = np.mean(joint_xcoords), np.std(joint_xcoords)
                    joint_ycoords_mean, joint_ycoords_std = np.mean(joint_ycoords), np.std(joint_ycoords)
                    joint_dxcoords_mean, joint_dxcoords_std = np.mean(joint_dxcoords), np.std(joint_dxcoords)
                    joint_dycoords_mean, joint_dycoords_std = np.mean(joint_dycoords), np.std(joint_dycoords) 

                    self.stats = {
                        'joint_xcoords_mean': joint_xcoords_mean, 'joint_xcoords_std': joint_xcoords_std,
                        'joint_ycoords_mean': joint_ycoords_mean, 'joint_ycoords_std': joint_ycoords_std,
                        'joint_dxcoords_mean': joint_dxcoords_mean, 'joint_dxcoords_std': joint_dxcoords_std,
                        'joint_dycoords_mean': joint_dycoords_mean, 'joint_dycoords_std': joint_dycoords_std
                    }
                    
                    
                os.makedirs(os.path.join(self.args.path, 'prepare_stats'), exist_ok=True)
                with open(os.path.join(self.args.path, 'prepare_stats', 'stats_train.pickle'), 'wb') as f:
                    pickle.dump(self.stats, f)
                    
                if self.args.horizontal_flip_augment and self.args.horizontal_flip_augment_purturb:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'horizontal_flip_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.horizontal_flip_augment_joint_randomness, f)
                        
                if self.args.horizontal_move_augment and self.args.horizontal_move_augment_purturb:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'horizontal_move_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.horizontal_move_augment_joint_randomness, f)
                        
                if self.args.vertical_move_augment and self.args.vertical_move_augment_purturb:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'vertical_move_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.vertical_move_augment_joint_randomness, f)
                        
                if self.args.agent_dropout_augment:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'agent_dropout_augment_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.agent_dropout_augment_randomness, f)
                    
            else:
                if self.logger is not None:
                    self.logger.info('Loading statistics for standardization...')
                try:
                    with open(os.path.join(self.args.path, 'prepare_stats', 'stats_train.pickle'), 'rb') as f:
                        self.stats = pickle.load(f)
                except FileNotFoundError:
                    print('Dataset statistics (e.g., mean, std) are missing! The dataset statistics pickle file should be generated during training.')
                    os._exit(0)
                    
                if self.args.horizontal_flip_augment and self.args.horizontal_flip_augment_purturb:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'horizontal_flip_augment_joint_randomness.pickle'), 'rb') as f:
                        self.horizontal_flip_augment_joint_randomness = pickle.load(f)
                        
                if self.args.horizontal_move_augment and self.args.horizontal_move_augment_purturb:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'horizontal_move_augment_joint_randomness.pickle'), 'rb') as f:
                        self.horizontal_move_augment_joint_randomness = pickle.load(f)
                        
                if self.args.vertical_move_augment and self.args.vertical_move_augment_purturb:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'vertical_move_augment_joint_randomness.pickle'), 'rb') as f:
                        self.vertical_move_augment_joint_randomness = pickle.load(f)
                
                if self.args.agent_dropout_augment:
                    with open(os.path.join(self.args.path, 'prepare_stats', 
                                           'agent_dropout_augment_randomness.pickle'), 'rb') as f:
                        self.agent_dropout_augment_randomness = pickle.load(f)
        else:
            try:
                with open(os.path.join(self.args.path, 'prepare_stats', 'stats_train.pickle'), 'rb') as f:
                    self.stats = pickle.load(f)
            except FileNotFoundError:
                print('Dataset statistics (e.g., mean, std) are missing! The dataset statistics pickle file should be generated during training.')
                os._exit(0)
                
                
    def read_ball_trajectory(self, filepath):
        with open(filepath , 'r') as f:
            ball_trajectory_lines = f.readlines()
        ball_trajectory = []
        for line in ball_trajectory_lines:
            x, y = line.rstrip().split()
            ball_trajectory.append([int(x), int(y)])
        return np.array(ball_trajectory)
            
    
    def joints_sanity_fix(self, joint_raw, frames):
        # note that it is possible the width_coords>1280 and height_coords>720 due to imperfect pose esitimation
        # here we fix these cases
        
        for t in joint_raw:
            for n in range(len(joint_raw[t])):
                for j in range(len(joint_raw[t][n])):
                    # joint_raw[t][n, j, 0] = int(joint_raw[t][n, j, 0])
                    # joint_raw[t][n, j, 1] = int(joint_raw[t][n, j, 1])
                    
                    if joint_raw[t][n, j, 0] >= self.args.image_w:
                        joint_raw[t][n, j, 0] = self.args.image_w - 1
                        
                    if joint_raw[t][n, j, 1] >= self.args.image_h:
                        joint_raw[t][n, j, 1] = self.args.image_h - 1
                    
                    if joint_raw[t][n, j, 0] < 0:
                        joint_raw[t][n, j, 0] = 0
                        
                    if joint_raw[t][n, j, 1] < 0:
                        joint_raw[t][n, j, 1] = 0 
                        
        # modify joint_raw - loop over each frame and pad the person dim because it can have less than N persons
        for f in joint_raw:
            n_persons = joint_raw[f].shape[0]
            if n_persons < self.args.N:  # padding in case some clips has less than N persons 
                joint_raw[f] = np.concatenate((
                    joint_raw[f], 
                    np.zeros((self.args.N-n_persons, self.args.J, joint_raw[f].shape[2]))), 
                    axis=0)
        return joint_raw
    
    
    def ball_trajectory_sanity_fix(self, ball_trajectory):
        # ball_trajectory: (T, 2)
        for t in range(len(ball_trajectory)):
            if ball_trajectory[t, 0] >= self.args.image_w:
                ball_trajectory[t, 0] = self.args.image_w - 1
                
            if ball_trajectory[t, 1] >= self.args.image_h:
                ball_trajectory[t, 1] = self.args.image_h - 1

            if ball_trajectory[t, 0] < 0:
                ball_trajectory[t, 0] = 0

            if ball_trajectory[t, 1] < 0:
                ball_trajectory[t, 1] = 0 
        return ball_trajectory

    def __getitem__(self, index):
        # index = 0
        current_joint_feats_path = self.clip_joints_paths[index] 
        (video, clip) = self.clips[index]
        label = self.annotations[index]
        person_labels = self.annotations_each_person[index]
        
        joint_raw = pickle.load(open(current_joint_feats_path, "rb"))
        # joint_raw: T: (N, J, 3)
        # 3: [joint_x, joint_y, joint_type]
        
        frames = sorted(joint_raw.keys())[self.args.frame_start_idx:self.args.frame_end_idx+1:self.args.frame_sampling]

        if self.args.ball_trajectory_use:
            ball_trajectory_data = self.read_ball_trajectory(self.clip_ball_paths[index])
            ball_trajectory_data = ball_trajectory_data[self.args.frame_start_idx+10:self.args.frame_end_idx+1+10:self.args.frame_sampling]
            # ball trajectory annotation has 41 frames annotated but joint/track file only has 20 frames.
            assert len(ball_trajectory_data) == len(frames)
            # (T, 2)
                        
        person_labels = torch.LongTensor(person_labels[frames[0]].squeeze())  # person action remains to be the same across all frames 
        # person_labels: (N, )
        
        # if horizontal flip augmentation and is training
        if self.args.horizontal_flip_augment and self.split == 'train':
            if index < len(self.horizontal_flip_mask):
                if self.horizontal_flip_mask[index]:
                    if self.args.horizontal_flip_augment_purturb:
                        joint_raw = horizontal_flip_augment_joint(
                            joint_raw, frames, self.horizontal_flip_augment_joint_randomness, self.args.image_w, add_purturbation=True, randomness_set=True, index=index)
                    else:
                        joint_raw = horizontal_flip_augment_joint(joint_raw, frames, self.horizontal_flip_augment_joint_randomness, self.args.image_w)
                    label = self.classidx_horizontal_flip_augment[label]  # label has to be flipped!
                    
                    if self.args.ball_trajectory_use:
                        ball_trajectory_data = horizontal_flip_ball_trajectory(ball_trajectory_data, self.args.image_w)
                        
        # if horizontal move augmentation and is training
        if self.args.horizontal_move_augment and self.split == 'train':
            if index < len(self.horizontal_mask):
                if self.horizontal_mask[index]:
                    if self.args.ball_trajectory_use:
                        if self.args.horizontal_move_augment_purturb:
                            joint_raw, ball_trajectory_data = horizontal_move_augment_joint(
                                joint_raw, frames, self.horizontal_move_augment_joint_randomness, add_purturbation=True, randomness_set=True, 
                                index=index, ball_trajectory=ball_trajectory_data)
                        else:
                            joint_raw, ball_trajectory_data = horizontal_move_augment_joint(
                                joint_raw, frames, self.horizontal_move_augment_joint_randomness, ball_trajectory=ball_trajectory_data)
                    else:
                        if self.args.horizontal_move_augment_purturb:
                            joint_raw = horizontal_move_augment_joint(
                                joint_raw, frames, self.horizontal_move_augment_joint_randomness, add_purturbation=True, randomness_set=True, index=index)
                        else:
                            joint_raw = horizontal_move_augment_joint(joint_raw, frames, self.horizontal_move_augment_joint_randomness)
                        
        # if vertical move augmentation and is training
        if self.args.vertical_move_augment and self.split == 'train':
            if index < len(self.vertical_mask):
                if self.vertical_mask[index]:
                    if self.args.ball_trajectory_use:
                        if self.args.vertical_move_augment_purturb:
                            joint_raw, ball_trajectory_data = vertical_move_augment_joint(
                                joint_raw, frames, self.vertical_move_augment_joint_randomness, add_purturbation=True,
                                randomness_set=True, index=index,
                                ball_trajectory=ball_trajectory_data)
                        else:
                            joint_raw, ball_trajectory_data = vertical_move_augment_joint(
                                joint_raw, frames, self.vertical_move_augment_joint_randomness, ball_trajectory=ball_trajectory_data)
                    else:
                        if self.args.vertical_move_augment_purturb:
                            joint_raw = vertical_move_augment_joint(
                                joint_raw, frames, self.vertical_move_augment_joint_randomness, add_purturbation=True,
                                randomness_set=True, index=index)
                        else:
                            joint_raw = vertical_move_augment_joint(joint_raw, frames, self.vertical_move_augment_joint_randomness)
         
        joint_raw = self.joints_sanity_fix(joint_raw, frames)
        if self.args.ball_trajectory_use:
            ball_trajectory_data = self.ball_trajectory_sanity_fix(ball_trajectory_data)
        
        
        # get joint_coords_all for image coordinates embdding
        if self.args.image_position_embedding_use:
            joint_coords_all = []
            for n in range(self.args.N):
                joint_coords_n = []

                for j in range(self.args.J):
                    joint_coords_j = []

                    for tidx, frame in enumerate(frames):
                        joint_x, joint_y, joint_type = joint_raw[frame][n,j,:]
                        
                        joint_x = min(joint_x, self.args.image_w-1)
                        joint_y = min(joint_y, self.args.image_h-1)
                        joint_x = max(0, joint_x)
                        joint_y = max(0, joint_y)

                        joint_coords = []
                        joint_coords.append(joint_x)  # width axis 
                        joint_coords.append(joint_y)  # height axis
                            
                        joint_coords_j.append(joint_coords)
                    joint_coords_n.append(joint_coords_j)   
                joint_coords_all.append(joint_coords_n)

        # get basic joint features (std normalization)
        joint_feats_basic = []  # (N, J, T, d_0_v1) 
        for n in range(self.args.N):
            joint_feats_n = []
            for j in range(self.args.J):
                joint_feats_j = []
                for tidx, frame in enumerate(frames):
                    joint_x, joint_y, joint_type = joint_raw[frame][n,j,:]

                    joint_feat = []

                    joint_feat.append((joint_x-self.stats['joint_xcoords_mean'])/self.stats['joint_xcoords_std'])
                    joint_feat.append((joint_y-self.stats['joint_ycoords_mean'])/self.stats['joint_ycoords_std'])

                    if tidx != 0:
                        pre_frame = frames[tidx-1] 
                        pre_joint_x, pre_joint_y, pre_joint_type = joint_raw[pre_frame][n,j,:]
                        joint_dx, joint_dy = joint_x - pre_joint_x, joint_y - pre_joint_y 
                    else:
                        joint_dx, joint_dy = 0, 0

                    joint_feat.append((joint_dx-self.stats['joint_dxcoords_mean'])/self.stats['joint_dxcoords_std'])
                    joint_feat.append((joint_dy-self.stats['joint_dycoords_mean'])/self.stats['joint_dycoords_std'])
                    joint_feats_j.append(joint_feat)
                joint_feats_n.append(joint_feats_j)
            joint_feats_basic.append(joint_feats_n)

        # person-wise normalization
        joint_feats_advanced = []  # (N, J, T, d_0_v2)

        joint_personwise_normalized = dict()
        for f in frames:
            joint_personwise_normalized[f] = personwise_normalization(
                torch.Tensor(joint_raw[f][:,:,:-1])) 

        for n in range(self.args.N):
            joint_feats_n = []

            for j in range(self.args.J):
                joint_feats_j = []

                for frame in frames:
                    joint_x, joint_y = joint_personwise_normalized[frame][n,j,:]
                    joint_type = joint_raw[frame][n,j,-1]

                    joint_feat = []
                    joint_feat.append(joint_x)
                    joint_feat.append(joint_y)
                    joint_feat.append(int(joint_type))  # start from 0

                    joint_feats_j.append(joint_feat)
                joint_feats_n.append(joint_feats_j)
            joint_feats_advanced.append(joint_feats_n)

        # adding joint metric features
        joint_feats_metrics = []  # (N, J, T, d_metrics)
        for frame_idx, frame in enumerate(frames):  # loop over frames of this clip
            this_frame_metric_scores = []
            for player_idx in range(self.args.N):  # loop over players
                this_player_metric_scores = []
                for joint_idx in range(self.args.J):  # loop over joints
                    if frame_idx == 0:  # first frame
                        this_player_metric_scores.append([1.0])
                    else:
                        frame_previous = frames[frame_idx-1]
                        OKS_score = oks_one_keypoint_compute(
                            joint_raw[frame][player_idx,joint_idx,:],
                            joint_raw[frame_previous][player_idx,joint_idx,:],
                            self.tdata[(int(video), int(clip))][frame][player_idx],
                            self.tdata[(int(video), int(clip))][frame_previous][player_idx]
                            )
                        this_player_metric_scores.append([OKS_score])
                this_frame_metric_scores.append(this_player_metric_scores)
            joint_feats_metrics.append(this_frame_metric_scores)
        joint_feats_metrics = np.array(joint_feats_metrics) # (T, N, J, 2) or  # (T, N, J, 1)

        # mean aggregate by a person's joints
        person_agg = np.mean(joint_feats_metrics, axis=2)
        joint_feats_metrics = np.concatenate(
            (joint_feats_metrics,
             np.tile(person_agg, self.args.J)[:,:,:,np.newaxis]), axis=-1)

        # (N, J, T, d)
        joint_feats = torch.cat((torch.Tensor(np.array(joint_feats_basic)),
                                 torch.Tensor(np.array(joint_feats_metrics)).permute(1,2,0,3),
                                 torch.Tensor(np.array(joint_feats_advanced)),
                                 torch.Tensor(np.array(joint_coords_all))), dim=-1)

        # if random agent dropout augmentation and is training                
        if self.args.agent_dropout_augment and self.split == 'train':
            if index < len(self.agent_dropout_mask):
                if self.agent_dropout_mask[index]:
                    joint_feats = agent_dropout_augment_joint(
                            joint_feats, frames, self.agent_dropout_augment_randomness[index])

        if self.args.ball_trajectory_use:
            # get ball trajectory features (std normalization)
            ball_feats_basic = []  # (T, 4)

            for t in range(len(ball_trajectory_data)):
                ball_x, ball_y = ball_trajectory_data[t]

                ball_feat = []

                ball_feat.append((ball_x-self.stats['ball_xcoords_mean'])/self.stats['ball_xcoords_std'])
                ball_feat.append((ball_y-self.stats['ball_ycoords_mean'])/self.stats['ball_ycoords_std'])

                if t != 0:
                    pre_ball_x, pre_ball_y  = ball_trajectory_data[t-1]
                    ball_dx, ball_dy = ball_x - pre_ball_x, ball_y - pre_ball_y 
                else:
                    ball_dx, ball_dy = 0, 0

                ball_feat.append((ball_dx-self.stats['ball_dxcoords_mean'])/self.stats['ball_dxcoords_std'])
                ball_feat.append((ball_dy-self.stats['ball_dycoords_mean'])/self.stats['ball_dycoords_std'])


                ball_feats_basic.append(ball_feat)
                
            ball_feats = torch.cat((torch.Tensor(np.array(ball_feats_basic)), torch.Tensor(ball_trajectory_data)), dim=-1)
            # (T, 6)
        else:
            ball_feats = torch.zeros(len(frames), 6)

        assert not torch.isnan(joint_feats).any()
        output = {
            'joint_feats': joint_feats,
            'ball_feats': ball_feats,
            'group_label': label,
            'person_labels': person_labels,
            'group_label_name': next(iter(self.idx2class[label])),
        }

        return output