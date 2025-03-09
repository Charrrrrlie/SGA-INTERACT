import os
import pickle
import numpy as np

from torch.utils.data import Dataset
from modules.mpgcn.graphs import Graph

from datasets.utils import graph_processing, multi_input
from datasets.constants import VOLLEYBALL_ACTION_DICT

class Volleyball_MPGCN(Dataset):
    def __init__(self, args, logger, split='train'):
        self.args = args
        self.split = split

        self.graph = Graph(**self.args.graph)

        data_path = os.path.join(args.path, split+'_data.npy')
        label_path = os.path.join(args.path, split+'_label.pkl')
        object_path = os.path.join(args.path, split+'_object_data.npy')

        self.idx2class = VOLLEYBALL_ACTION_DICT

        if logger is not None:
            logger.info('Loading {} pose data from {}'.format(split, data_path))
        self.data = np.load(data_path)
        # N, T, M, V, C -> N, C, T, V, M
        self.data = self.data.transpose(0, 4, 1, 3, 2)

        with open(label_path, 'rb') as f:
            self.label = pickle.load(f)

        if args.ball_trajectory_use:
            if logger is not None:
                logger.info('Loading {} object data from {}'.format(split, object_path))
            self.object_data = np.load(object_path)
            # (N, T, v, C) -> (N, C, T, v)
            self.object_data = self.object_data.transpose(0, 3, 1, 2)

            # (N, C, T, v) -> (N, C, T, v, M)
            self.object_data = np.expand_dims(self.object_data, axis=-1)
            self.object_data = np.tile(self.object_data, (1, 1, 1, 1, self.args.N))

            # (N, C, T, V, M) -> (N, C, T, V+v, M)
            self.data = np.concatenate((self.data, self.object_data), axis = 3)

        self.data = self.data[:, :self.args.input_dims, range(*self.args.window), :, :]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # (C, T, V, M)
        pose_data = np.array(self.data[idx])
        label, clip_name = self.label[idx]

        pose_data = graph_processing(pose_data, self.graph.form, self.args.graph.processing)
        joint_feats = multi_input(pose_data, self.graph.connect_joint, self.args.inputs, self.graph.center)

        output = {
            'joint_feats': joint_feats.astype(np.float32),
            'group_label': label,
            'group_label_name': next(iter(self.idx2class[label])),
        }
        return output
