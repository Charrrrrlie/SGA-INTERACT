import os
import glob
import pickle
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from datasets.constants import BASKETBALL_ACTION_SET, COCO_KEYPOINT_HORIZONTAL_FLIPPED
from datasets.utils import get_djoints
from datasets.utils import flip_augment_joint3D, move_augment_joint3D, agent_dropout_augment_joint3D, agent_temporal_augment3D

# composer utils
from datasets.utils import personwise_normalization, oks_keypoint_3d_compute

# mpgcn utils
from modules.mpgcn.graphs import Graph
from datasets.utils import graph_processing, multi_input

# stgcn utils
from datasets.utils import COCO_to_OpenPose25

# not enough cases for some actions
IGNORE_ACTIONS = ['Baseline Cut', 'Through', 'Shuffle']

def preprocess_joints(player_joints):
    """
        padding static pose from the end (out of court) as zeros
    """
    clip_len = -1
    for player, joints in player_joints.items():
        if clip_len == -1:
            clip_len = len(joints)
        last_valid_idx = len(joints) - 1
        for i in range(len(joints) - 1, 1):
            if not np.allclose(joints[i], joints[i - 1]):
                last_valid_idx = i
                break
        joints[last_valid_idx + 1:] = np.zeros_like(joints[last_valid_idx + 1:])
        player_joints[player] = joints

    return clip_len, player_joints

class BasketballGAR(Dataset):
    """
        Basketball dataset for Group Action Recognition
        T, N, J, C denotes time, player num, joint num and channel num respectively
    """
    def __init__(self, args, logger, split='train'):
        self.args = args

        # padding parameter
        if 'force_T' not in self.args:
            self.args.force_T = False

        self.logger = logger

        self.split = split

        self.cal_stats_info()
        action_split_dict = self.create_split_info()

        self.action2id = {}
        count = 0
        for action in sorted(BASKETBALL_ACTION_SET):
            if action not in IGNORE_ACTIONS:
                self.action2id[action] = count
                count += 1

        self.id2action = {v: k for k, v in self.action2id.items()}

        # organize the split info
        self.base_info_list = self.create_info_list(action_split_dict, IGNORE_ACTIONS)
        if self.logger:
            self.logger.info(f'{len(self.base_info_list)} samples are loaded.')

        if self.args.output_type == 'mpgcn':
            self.graph = Graph(**self.args.graph)
        else:
            self.graph = None

        # keypoints info
        if self.args.output_type == 'composer':
            self.coco_index = []
            self.coco_flipped_index = []
            for i in range(self.args.J):
                self.coco_index.append(i)
                self.coco_flipped_index.append(COCO_KEYPOINT_HORIZONTAL_FLIPPED[i])

        # augmentation
        self.aug_func_list = []
        self.aug_func_idx = {}
        self.ori_data_len = len(self.base_info_list)
        if 'aug' in self.args and split == 'train':
            for i, aug in enumerate(self.args.aug.aug_func):
                self.aug_func_list.append(eval(aug))
                if aug not in self.aug_func_idx:
                    # recording start idx to process repeated augmentation functions
                    self.aug_func_idx[aug] = i
            self.base_info_list = self.base_info_list * (len(self.aug_func_list) + 1)

    def create_info_list(self, action_split_dict, IGNORE_ACTIONS):
        """
            convert action_split_dict in {'action_name': [(path, action_idx), ...]}
            to
            info_list in [(path, action_idx, action_name), ...]
        """
        base_info_list = []
        for key, val in action_split_dict.items():
            if key in IGNORE_ACTIONS:
                continue
            val = [(p, i, key) for p, i in val]
            base_info_list.extend(val)
        return base_info_list

    def logging_info(self, action_split_dict):
        text = ''
        for key, val in action_split_dict.items():
            text += f'Num {key}: {len(val)}; '
        self.logger.info('Split info: ' + text)

    def create_split_info(self, split_ratio=0.3):
        """
            returns dict: {action_name: [(os.path.basename(tactic_annot_file1), action_idx), ...]}
        """
        if os.path.exists(os.path.join(self.args.path, f'GAR_{self.split}_split_{split_ratio}ratio_info.pkl')):
            if self.logger:
                self.logger.info('Loading split info from files...')
            with open(os.path.join(self.args.path, f'GAR_{self.split}_split_{split_ratio}ratio_info.pkl'), 'rb') as f:
                action_split_dict = pickle.load(f)
            if self.logger:
                self.logging_info(action_split_dict)
            return action_split_dict
        if self.logger:
            self.logger.info('Creating split info...')
            self.logger.info('Summarize action info...')
        if os.path.exists(os.path.join(self.args.path, 'GAR_action_info.pkl')):
            with open(os.path.join(self.args.path, 'GAR_action_info.pkl'), 'rb') as f:
                action_dict = pickle.load(f)
        else:
            action_dict = defaultdict(list)
            action_annot_file_list = glob.glob(os.path.join(self.args.path, 'annots/tactic', '*.pkl'))
            for action_annot_file in action_annot_file_list:
                with open(action_annot_file, 'rb') as f:
                    action_annot = pickle.load(f)
                action_annot = action_annot['Action']
                for key, value in action_annot.items():
                    assert key in BASKETBALL_ACTION_SET, f'Invalid action name {key}'
                    for i, val in enumerate(value):
                        action_dict[key].append((os.path.basename(action_annot_file), i))
            with open(os.path.join(self.args.path, 'GAR_action_info.pkl'), 'wb') as f:
                pickle.dump(action_dict, f)

        if self.logger:
            self.logger.info('Sampling train & test set according to action info...')
        for key in action_dict.keys():
            np.random.shuffle(action_dict[key])

        action_test_dict = defaultdict(list)
        action_train_dict = defaultdict(list)

        for key in action_dict.keys():
            num_test = max(1, int(len(action_dict[key]) * split_ratio))
            action_test_dict[key] = action_dict[key][:num_test]
            action_train_dict[key] = action_dict[key][num_test:]

        with open(os.path.join(self.args.path, f'GAR_train_split_{split_ratio}ratio_info.pkl'), 'wb') as f:
            pickle.dump(action_train_dict, f)
        with open(os.path.join(self.args.path, f'GAR_test_split_{split_ratio}ratio_info.pkl'), 'wb') as f:
            pickle.dump(action_test_dict, f)

        action_split_dict = action_train_dict if self.split == 'train' else action_test_dict
        if self.logger:
            self.logging_info(action_split_dict)
        return action_split_dict

    def cal_stats_info(self):
        """
            calculate the mean, std info of joints
        """
        if os.path.exists(os.path.join(self.args.path, 'GAR_joint_stats.pkl')):
            if self.logger:
                self.logger.info('Loading mean, std info of joints...')
            with open(os.path.join(self.args.path, 'GAR_joint_stats.pkl'), 'rb') as f:
                stats_dict = pickle.load(f)
                self.mean_joint = stats_dict['mean_joint']
                self.std_joint = stats_dict['std_joint']
                self.mean_djoint = stats_dict['mean_djoint']
                self.std_djoint = stats_dict['std_djoint']
        else:
            joint_path_list = glob.glob(os.path.join(self.args.path, 'joints', '*.npy'))
            self.mean_joint = np.zeros((3))
            self.std_joint = np.zeros((3))
            self.mean_djoint = np.zeros((3))
            self.std_djoint = np.zeros((3))

            total_time_len = 0
            if self.logger:
                self.logger.info('Calculating mean, std info of joints...')
            for j_l in joint_path_list:
                player_joints = np.load(j_l, allow_pickle=True).item()
                for player, joints in player_joints.items():
                    total_time_len += len(joints)
                    break
            for j_l in joint_path_list:
                player_joints = np.load(j_l, allow_pickle=True).item()
                for player, joints in player_joints.items():
                    djoints = get_djoints(joints)
                    self.mean_joint += np.mean(joints, axis=(0, 1)) * len(joints) / total_time_len / self.args.N
                    self.std_joint += np.std(joints, axis=(0, 1)) * len(joints) / total_time_len / self.args.N
                    self.mean_djoint += np.mean(djoints, axis=(0, 1)) * len(joints) / total_time_len / self.args.N
                    self.std_djoint += np.std(djoints, axis=(0, 1)) * len(joints) / total_time_len / self.args.N

            with open(os.path.join(self.args.path, 'GAR_joint_stats.pkl'), 'wb') as f:
                pickle.dump({'mean_joint': self.mean_joint,
                            'std_joint': self.std_joint,
                            'mean_djoint': self.mean_djoint,
                            'std_djoint': self.std_djoint}, f)
        if self.logger:
            self.logger.info(f'Mean joint: {self.mean_joint}, std joint: {self.std_joint}')
            self.logger.info(f'Mean djoint: {self.mean_djoint}, std djoint: {self.std_djoint}')
        return

    def split_action_clip(self, base_path, player_traj, ball_traj, action_name, action_idx):
        with open(os.path.join(self.args.path, 'annots/tactic', base_path), 'rb') as f:
            tactic_annot = pickle.load(f)
        tactic_annot = tactic_annot['Action'][action_name][action_idx]
        round_len = player_traj.shape[0]

        clip_start_frame = round(tactic_annot[0] * self.args.fps)
        clip_end_frame = min(round(tactic_annot[1] * self.args.fps) + 1, round_len)
        player_traj = player_traj[clip_start_frame: clip_end_frame]
        ball_traj = ball_traj[clip_start_frame: clip_end_frame]

        return player_traj, ball_traj

    def COMPOSER_output(self, player_traj, ball_traj, index):
        """
            formulate COMPOSER output
            input:
                player_traj: [T, N, J, 3]
                ball_traj: [T, N]
                index: int, only used for aug
        """
        # ----------------- aug information -----------------
        if 'aug' in self.args and self.split == 'train' and index >= self.ori_data_len:
            use_aug = True
            aug_idx = index // self.ori_data_len - 1
            aug_func = self.aug_func_list[aug_idx]
            aug_func_name = self.args.aug.aug_func[aug_idx]
        else:
            use_aug = False

        # ----------------- all joints -----------------
        joint_coords_all = player_traj.copy()
        joint_coords_all[..., 0] = np.clip(joint_coords_all[..., 0], 0, 11 - 0.001)
        joint_coords_all[..., 1] = np.clip(joint_coords_all[..., 1], 0, 15 - 0.001)

        # ----------------- basic features -----------------
        player_traj_basic = player_traj.copy()
        player_dtraj_basic = get_djoints(player_traj_basic)
        player_traj_basic = (player_traj_basic - self.mean_joint) / self.std_joint
        player_dtraj_basic = (player_dtraj_basic - self.mean_djoint) / self.std_djoint

        joint_feats_basic = np.concatenate([player_traj_basic, player_dtraj_basic], axis=-1)

        # ----------------- advanced features -----------------
        player_traj_advance = torch.tensor(player_traj.copy())
        for t in range(len(player_traj_advance)):
            player_traj_advance[t] = personwise_normalization(player_traj_advance[t])

        if use_aug and aug_func_name == 'flip_augment_joint3D':
            kp_type = self.coco_flipped_index
        else:
            kp_type = self.coco_index
        player_joint_type = torch.tensor(kp_type).to(torch.float32)\
                                .reshape([1, 1, -1, 1]).repeat(player_traj.shape[0], player_traj.shape[1], 1, 1)
        joint_feats_advanced = torch.cat([player_traj_advance, player_joint_type], dim=-1).numpy()

        # ----------------- oks features -----------------
        full_oks = oks_keypoint_3d_compute(player_traj.copy())
        person_oks = np.mean(full_oks, axis=2, keepdims=True).repeat(full_oks.shape[2], axis=2)
        joint_feats_metrics = np.concatenate([full_oks, person_oks], axis=-1)

        # [T, N, J, C]
        # [T, N, J, 6] + [T, N, J, 2] + [T, N, J, 4] + [T, N, J, 3]
        joint_feats = np.concatenate([joint_feats_basic,
                                      joint_feats_metrics,
                                      joint_feats_advanced,
                                      joint_coords_all], axis=-1)

        # augmentation
        if use_aug and aug_func_name == 'agent_dropout_augment_joint3D':
            joint_feats = aug_func(joint_feats, **self.args.aug.aug_param[aug_func_name])

        clip_len = joint_feats.shape[0]

        # padding
        assert joint_feats.shape[0] == ball_traj.shape[0], 'Invalid joint_feats / ball_traj shape'

        if not self.args.force_T or self.args.T > clip_len:
            assert self.args.T >= clip_len, f'Padding size must be larger than the round length with {self.args.T} and {clip_len}'

            joint_feats = np.pad(joint_feats,
                                ((0, self.args.T - clip_len), (0, 0), (0, 0), (0, 0)),
                                mode='constant',
                                constant_values=0)
            ball_traj = np.pad(ball_traj,
                            ((0, self.args.T - clip_len), (0, 0)),
                            mode='constant',
                            constant_values=0)
        else:
            # NOTE(yyc): ONLY used for GAR.
            # Since only one case is longer than 400 frames, extra padding will affect the performance.
            start_idx = (joint_feats.shape[0] - self.args.T) // 2
            joint_feats = joint_feats[start_idx: start_idx + self.args.T]
            ball_traj = ball_traj[start_idx: start_idx + self.args.T]

        # convert [T, N, J, C] to [N, J, T, C]
        joint_feats = np.transpose(joint_feats, (1, 2, 0, 3))

        output = {
            'joint_feats': joint_feats.astype(np.float32),
        }

        if self.args.ball_trajectory_use:
            output['ball_occupy_feats'] = ball_traj.astype(np.float32)

        return output

    def MPGCN_output(self, player_traj, ball_traj, index):
        """
            formulate MPGCN output. Note: random dropout augmentation is not used in MPGCN.
            input:
                player_traj: [T, N, J, 3]
                ball_traj: [T, N]
                index: int, only used for aug
            output:
                joint_feats: [I, C, T, V, M],
                    denotes len(self.inputs) / different joint representaion, 2 * joint_dim, T, graph node num, group num
        """

        player_traj = player_traj.transpose(3, 0, 2, 1)
        player_traj = graph_processing(player_traj, self.graph.form, self.args.graph.processing)
        joint_feats = multi_input(player_traj, self.graph.connect_joint, self.args.inputs, self.graph.center)

        I, C, T, V, M = joint_feats.shape
        if not self.args.force_T or self.args.T >= T:
            assert self.args.T >= T, f'Padding size must be larger than the round length with {self.args.T} and {T}'
            joint_feats = np.pad(joint_feats,
                                ((0, 0), (0, 0), (0, self.args.T - T), (0, 0), (0, 0)),
                                mode='constant',
                                constant_values=0)
            ball_traj = np.pad(ball_traj,
                            ((0, self.args.T - T), (0, 0)),
                            mode='constant',
                            constant_values=0)
        else:
            # NOTE(yyc): ONLY used for GAR.
            # Since only one case is longer than 400 frames, extra padding will affect the performance.
            start_idx = (T - self.args.T) // 2
            joint_feats = joint_feats[:, :, start_idx: start_idx + self.args.T, ...]
            ball_traj = ball_traj[start_idx: start_idx + self.args.T, ...]

        output = {
            'joint_feats': joint_feats.astype(np.float32),
        }
        if self.args.ball_trajectory_use:
            output['ball_occupy_feats'] = ball_traj.astype(np.float32)

        return output

    def STGCN_output(self, player_traj, ball_traj, index):
        """
            formulate COMPOSER output
            input:
                player_traj: [T, N, J, 3]
                ball_traj: [T, N]
                index: int, only used for aug
            output:

        """
        # ----------------- aug information -----------------
        if 'aug' in self.args and self.split == 'train' and index >= self.ori_data_len:
            use_aug = True
            aug_idx = index // self.ori_data_len - 1
            aug_func = self.aug_func_list[aug_idx]
            aug_func_name = self.args.aug.aug_func[aug_idx]
        else:
            use_aug = False

        if 'pre_norm' in self.args and self.args.pre_norm:
            player_traj = (player_traj - self.mean_joint) / self.std_joint

        joint_feats = COCO_to_OpenPose25(player_traj)

        if use_aug and aug_func_name == 'agent_dropout_augment_joint3D':
            joint_feats = aug_func(joint_feats, **self.args.aug.aug_param[aug_func_name])

        T, N, J, _ = player_traj.shape
        if not self.args.force_T or self.args.T >= T:
            assert self.args.T >= T, f'Padding size must be larger than the round length with {self.args.T} and {T}'
            joint_feats = np.pad(joint_feats,
                                ((0, self.args.T - T), (0, 0), (0, 0), (0, 0)),
                                mode='constant',
                                constant_values=0)
            ball_traj = np.pad(ball_traj,
                            ((0, self.args.T - T), (0, 0)),
                            mode='constant',
                            constant_values=0)
        else:
            # NOTE(yyc): ONLY used for GAR.
            # Since only one case is longer than 400 frames, extra padding will affect the performance.
            start_idx = (T - self.args.T) // 2
            joint_feats = joint_feats[start_idx: start_idx + self.args.T]
            ball_traj = ball_traj[start_idx: start_idx + self.args.T]

        # [T, N, J, C] to [N, C, T, J]
        joint_feats = joint_feats.transpose(1, 3, 0, 2)

        output = {
            'joint_feats': joint_feats.astype(np.float32),
        }
        if self.args.ball_trajectory_use:
            output['ball_occupy_feats'] = ball_traj.astype(np.float32)

        return output

    def __getitem__(self, index):
        # same action may occur multiple times in the same round, thus we use action_idx to distinguish them
        base_path, action_idx, action_name = self.base_info_list[index]

        player_joints = np.load(os.path.join(self.args.path, 'joints',
                                              base_path.replace('_tactic.pkl', '_pose.npy')),
                                allow_pickle=True).item()
        assert len(player_joints) == self.args.N, 'Invalid player number'

        round_len, player_joints = preprocess_joints(player_joints)

        # get player_traj in [T, N, J, 3]
        # get binary ball_traj in [T, N] where 1 indicates the player occupies the ball
        with open(os.path.join(self.args.path, 'annots/ball', base_path.replace('_tactic.pkl', '_ball_traj.pkl')), 'rb') as f:
            ball_annot = pickle.load(f)

        player_traj = np.zeros((round_len, self.args.N, self.args.J, 3))
        ball_traj = np.zeros((round_len, self.args.N))
        host_team_info = []
        guest_team_info = []

        for idx, (player, joints) in enumerate(player_joints.items()):
            player_traj[:, idx, :] = joints
            if 'host' in player:
                host_team_info.append(idx)
            else:
                guest_team_info.append(idx)
            if player in ball_annot:
                for occupy_time in ball_annot[player]:
                    start_frame_idx = round(occupy_time[0] * self.args.fps)
                    end_frame_idx = min(round(occupy_time[1] * self.args.fps) + 1, round_len)
                    ball_traj[start_frame_idx: end_frame_idx, idx] = 1

        player_traj, ball_traj = self.split_action_clip(base_path, player_traj, ball_traj, action_name, action_idx)

        # player keypoint height must be positive
        player_traj[..., 2] = np.clip(player_traj[..., 2], 0, None)

        label = self.action2id[action_name]

        # augmentation
        if 'aug' in self.args and self.split == 'train' and index >= self.ori_data_len:
            aug_idx = index // self.ori_data_len - 1
            aug_func = self.aug_func_list[aug_idx]
            aug_func_name = self.args.aug['aug_func'][aug_idx]

            if aug_func_name != 'agent_dropout_augment_joint3D':
                aug_param = self.args.aug.aug_param[aug_func_name]

                if isinstance(next(iter(aug_param.values())), list):
                    _aug_param_idx = aug_idx - self.aug_func_idx[aug_func_name]
                    aug_param = {k: v[_aug_param_idx] for k, v in aug_param.items()}
                if aug_func_name == 'agent_temporal_augment3D':
                    player_traj, selected_frame = aug_func(player_traj, **aug_param)
                    ball_traj = ball_traj[selected_frame]
                else:
                    player_traj = aug_func(player_traj, **aug_param)

        assert self.args.output_type in ['composer', 'mpgcn', 'stgcn'], 'Invalid output type'
        output = getattr(self, f'{self.args.output_type.upper()}_output')(player_traj, ball_traj, index)

        output['group_label'] = label
        output['group_label_name'] = self.id2action[label]

        # we set offensive team as host team for coherence of team info embedding
        # with open(os.path.join(self.args.path, 'annots/tactic', base_path), 'rb') as f:
        #     annot = pickle.load(f)

        # if annot['Offensive'] == 'Blue':
        #     output['host_team_info'] = np.array(guest_team_info)
        #     output['guest_team_info'] = np.array(host_team_info)
        # else:
        output['host_team_info'] = np.array(host_team_info)
        output['guest_team_info'] = np.array(guest_team_info)

        return output

    def __len__(self):
        return len(self.base_info_list)

    def get_flops_demo_data(self, bs):

        player_traj = np.random.randn(self.args.T, self.args.N, self.args.J, 3)
        ball_traj = np.zeros((self.args.T, self.args.N))

        assert self.args.output_type in ['composer', 'mpgcn', 'stgcn'], 'Invalid output type'
        output = getattr(self, f'{self.args.output_type.upper()}_output')(player_traj, ball_traj, 0)
        output['group_label'] = torch.ones([1], dtype=torch.long).to('cuda:0')
        output['host_team_info'] = torch.tensor([0, 1, 2], dtype=torch.long).view(1, -1).to('cuda:0')
        output['guest_team_info'] = torch.tensor([3, 4, 5], dtype=torch.long).view(1, -1).to('cuda:0')

        output['joint_feats'] = torch.tensor(output['joint_feats']).unsqueeze(0).to('cuda:0')
        if 'ball_occupy_feats' in output:
            output['ball_occupy_feats'] = torch.tensor(output['ball_occupy_feats']).unsqueeze(0).to('cuda:0')

        for key, value in output.items():
            dim = value.dim()
            output[key] = value.repeat([bs] + [1] * (dim - 1))
        return output

class BasketballGAL(BasketballGAR):
    """
        Basketball dataset for Temporal Group Action Localization
    """
    def __init__(self, args, logger, split='train'):
        super(BasketballGAL, self).__init__(args, logger, split)
        self.preprocess_none_action_clips()

    def preprocess_none_action_clips(self):
        base_info_list_update = []
        for info_path in self.base_info_list:
            with open(os.path.join(self.args.path, 'annots/tactic', info_path), 'rb') as f:
                tactic_annot = pickle.load(f)
            tactic_annot = tactic_annot['Action']
            action_num = 0
            for key, value in tactic_annot.items():
                if key not in IGNORE_ACTIONS:
                    action_num += len(value)
                    break
            if action_num != 0:
                base_info_list_update.append(info_path)

        if self.logger:
            self.logger.info(f'{len(base_info_list_update)} samples are left after removing none action clips.')
        self.base_info_list = base_info_list_update

    def create_info_list(self, action_split_list, IGNORE_ACTIONS):
        """
            return list: [path1, ...]
        """
        return action_split_list

    def logging_info(self, action_set):
        pass

    def create_split_info(self, split_ratio=0.3):
        """
            returns list: [os.path.basename(annot_file1), ...]
        """
        if os.path.exists(os.path.join(self.args.path, f'GAL_{self.split}_split_{split_ratio}ratio_info.pkl')) \
            and os.path.exists(os.path.join(self.args.path, f'GAL_{self.split}_split_{split_ratio}ratio_info.pkl')):
            if self.logger:
                self.logger.info('Loading split info from files...')
            with open(os.path.join(self.args.path, f'GAL_{self.split}_split_{split_ratio}ratio_info.pkl'), 'rb') as f:
                action_train_set = pickle.load(f)
            with open(os.path.join(self.args.path, f'GAL_{self.split}_split_{split_ratio}ratio_info.pkl'), 'rb') as f:
                action_test_set = pickle.load(f)
            action_set = action_train_set if self.split == 'train' else action_test_set
            if self.logger:
                self.logging_info(action_set)
            return action_set

        if self.logger:
            self.logger.info('Creating split info...')
        action_annot_file_list = glob.glob(os.path.join(self.args.path, 'annots/tactic', '*.pkl'))
        np.random.shuffle(action_annot_file_list)
        action_annot_file_list = [os.path.basename(f) for f in action_annot_file_list]

        action_test_set = action_annot_file_list[:int(len(action_annot_file_list) * split_ratio)]
        action_train_set = action_annot_file_list[int(len(action_annot_file_list) * split_ratio):]

        with open(os.path.join(self.args.path, f'GAL_train_split_{split_ratio}ratio_info.pkl'), 'wb') as f:
            pickle.dump(action_train_set, f)
        with open(os.path.join(self.args.path, f'GAL_test_split_{split_ratio}ratio_info.pkl'), 'wb') as f:
            pickle.dump(action_test_set, f)

        action_set = action_train_set if self.split == 'train' else action_test_set
        if self.logger:
            self.logging_info(action_set)
        return action_set

    def __getitem__(self, index):
        info_path = self.base_info_list[index]

        player_joints = np.load(os.path.join(self.args.path, 'joints', info_path.replace('_tactic.pkl', '_pose.npy')), allow_pickle=True).item()
        assert len(player_joints) == self.args.N, 'Invalid player number'

        round_len, player_joints = preprocess_joints(player_joints)

        # get player_traj in [T, N, J, 3]
        # get binary ball_traj in [T, N] where 1 indicates the player occupies the ball
        with open(os.path.join(self.args.path, 'annots/ball', info_path.replace('_tactic.pkl', '_ball_traj.pkl')), 'rb') as f:
            ball_annot = pickle.load(f)

        player_traj = np.zeros((round_len, self.args.N, self.args.J, 3))
        ball_traj = np.zeros((round_len, self.args.N))
        host_team_info = []
        guest_team_info = []

        for idx, (player, joints) in enumerate(player_joints.items()):
            player_traj[:, idx, :] = joints
            if 'host' in player:
                host_team_info.append(idx)
            else:
                guest_team_info.append(idx)
            if player in ball_annot:
                for occupy_time in ball_annot[player]:
                    start_frame_idx = round(occupy_time[0] * self.args.fps)
                    end_frame_idx = min(round(occupy_time[1] * self.args.fps) + 1, round_len)
                    ball_traj[start_frame_idx: end_frame_idx, idx] = 1

        # player keypoint height must be positive
        player_traj[..., 2] = np.clip(player_traj[..., 2], 0.0, None)

        # NOTE(yyc): different from GAR, we do not split the action clip
        # and process label in a different way

        with open(os.path.join(self.args.path, 'annots/tactic', info_path), 'rb') as f:
            annot = pickle.load(f)
        tactic_annot = annot['Action']
        segment = []
        label = []
        label_names = []
        for key, value in tactic_annot.items():
            if key in IGNORE_ACTIONS:
                continue
            for i, val in enumerate(value):
                segment.append([val[0] * self.args.fps, val[1] * self.args.fps])
                label.append(self.action2id[key])
                label_names.append(key)

        # augmentation
        if 'aug' in self.args and self.split == 'train' and index >= self.ori_data_len:
            aug_idx = index // self.ori_data_len - 1
            aug_func = self.aug_func_list[aug_idx]
            aug_func_name = self.args.aug['aug_func'][aug_idx]

            if aug_func_name != 'agent_dropout_augment_joint3D':
                aug_param = self.args.aug.aug_param[aug_func_name]

                if isinstance(next(iter(aug_param.values())), list):
                    _aug_param_idx = aug_idx - self.aug_func_idx[aug_func_name]
                    aug_param = {k: v[_aug_param_idx] for k, v in aug_param.items()}
                if aug_func_name == 'agent_temporal_augment3D':
                    player_traj, selected_frame = aug_func(player_traj, **aug_param)
                    ball_traj = ball_traj[selected_frame]
                else:
                    player_traj = aug_func(player_traj, **aug_param)

        assert self.args.output_type in ['composer', 'mpgcn', 'stgcn'], 'Invalid output type'
        output = getattr(self, f'{self.args.output_type.upper()}_output')(player_traj, ball_traj, index)

        output['group_segment'] = np.array(segment)
        output['group_label'] = np.array(label)
        output['group_label_name'] = np.array(label_names)

        # padding
        current_action_num = len(segment)
        if current_action_num:
            output['group_segment'] = np.pad(output['group_segment'], ((0, self.args.max_action_num - current_action_num), (0, 0)), mode='constant', constant_values=0)
            output['group_label'] = np.pad(output['group_label'], (0, self.args.max_action_num - current_action_num), mode='constant', constant_values=-1)
            output['group_label_name'] = np.pad(output['group_label_name'], (0, self.args.max_action_num - current_action_num), mode='constant', constant_values='')

            output['group_label_name'] = list(output['group_label_name'])
        else:
            output['group_segment'] = np.zeros((self.args.max_action_num, 2))
            output['group_label'] = np.ones((self.args.max_action_num), dtype=np.int64) * -1
            output['group_label_name'] = [''] * self.args.max_action_num

        # we set offensive team as host team for coherence of team info embedding
        # if annot['Offensive'] == 'Blue':
        #     output['host_team_info'] = np.array(guest_team_info)
        #     output['guest_team_info'] = np.array(host_team_info)
        # else:
        output['host_team_info'] = np.array(host_team_info)
        output['guest_team_info'] = np.array(guest_team_info)

        output['video_id'] = index

        return output
