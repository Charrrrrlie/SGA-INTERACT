import os
import pickle
import numpy as np
from tqdm import tqdm

from datasets.constants import VOLLEYBALL_ACTION_DICT

class VolleyBall_Reader():
    def __init__(self, dataset_root_folder, out_folder, olympic_split, **kwargs):
        self.max_channel = 3
        self.max_frame = 20
        self.max_joint = 17
        self.max_person = 12

        self.dataset_root_folder = dataset_root_folder
        self.out_folder = out_folder

        # Divide train and eval samples
        if olympic_split:
            self.out_folder += '_olympic'
            self.training_samples = [1, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                          41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
            self.eval_samples = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                         19, 20, 21, 22, 23, 24, 25, 26, 27]
        else:
            self.training_samples = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39,
                                        40, 41, 42, 48, 50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26,
                                        27, 28, 30, 33, 46, 49, 51]
            self.eval_samples = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

        # Create label-to-idx map
        # self.class2idx = {
        #     'r_set': 0, 'r-set': 0, 'l_set': 1, 'l-set': 1,
        #     'r_spike': 2, 'r-spike': 2, 'l_spike': 3, 'l-spike': 3,
        #     'r_pass': 4, 'r-pass': 4, 'l_pass': 5, 'l-pass': 5,
        #     'r_winpoint': 6, 'r-winpoint': 6, 'l_winpoint': 7, 'l-winpoint': 7
        # }
        self.class2idx = {}
        for key, value in VOLLEYBALL_ACTION_DICT.items():
            for v in value:
                self.class2idx[v] = key

    def read_pose(self, joint_path, ball_path, clip):
        # joint_raw: T: (M, V, 3) 20: (12, 17, 3)
        # 3: [joint_x, joint_y, joint_type]
        # ball_raw: [2,41]

        C, T, V, M = 3, 41, 17, 12
        skeleton_data = np.zeros([T, M, V, C])
        
        # Read skeleton data
        with open(os.path.join(joint_path, clip+'.pickle'), 'rb') as f:
            joint_raw = pickle.load(f)
        for t in joint_raw.keys():
            skeleton_data[t-int(clip)+20] = joint_raw[t]

        # Read ball data
        with open(os.path.join(ball_path, clip+'.txt'), 'r') as f:
            ball_data = f.readlines()
        # T:41, C:2+1
        ball_data = np.array(
            [list(map(int, t.strip().split())) + [0] for t in ball_data])
        # T,C -> T,1,C
        ball_data = np.expand_dims(ball_data, axis=1)
        
        # only select middle 20 frames
        # ball_data = ball_data[10:30, :]
        # # T,C -> C,T,M
        # ball_data = np.expand_dims(ball_data.transpose(
        #     1, 0), axis=-1).repeat(M, axis=-1)
        # # C,T,M -> C,T,1,M
        # ball_data = np.expand_dims(ball_data, axis=2)

        return skeleton_data, ball_data

    def gendata(self, phase):

        res_skeleton = []
        res_ball = []
        group_labels = []
        individual_labels = []
        videos = self.training_samples if phase == 'train' else self.eval_samples

        # actions[(video_id, clip)][frame]: [N, 1]
        individual_label_path = os.path.join(
            self.dataset_root_folder, 'tracks_normalized_with_person_action_label.pkl')
        individual_label_dict = np.load(
            individual_label_path, allow_pickle=True)

        iterizer = tqdm(videos, dynamic_ncols=True)
        for video_id in iterizer:
            video_path = os.path.join(
                self.dataset_root_folder, 'videos', str(video_id))
            joint_path = os.path.join(
                self.dataset_root_folder, 'joints', str(video_id))
            ball_path = os.path.join(
                self.dataset_root_folder, 'volleyball_ball_annotation', str(video_id))
            # Get annotations
            with open(os.path.join(video_path, 'annotations.txt'), 'r') as f:
                lines = f.readlines()
            for l in lines:
                # Get clip id and group label
                clip, group_label = l.split()[0].split('.jpg')[0], l.split()[1]
                name = str(video_id) + '-' + clip
                group_labels.append([self.class2idx[group_label], name])

                # Get individual label and transpose [N, 1] -> [N]
                individual_label = individual_label_dict[(
                    video_id, int(clip))][int(clip)]
                individual_label = [[int(label), name + '-' + str(i)] for i, label in enumerate(individual_label)]
                individual_labels.extend(individual_label)

                # Get joint/+ball information
                joint_data, object_data = self.read_pose(joint_path, ball_path, clip)
                res_skeleton.append(joint_data)
                res_ball.append(object_data)

        # Save label
        os.makedirs(self.out_folder, exist_ok=True)
        with open(os.path.join(self.out_folder, phase + '_label.pkl'), 'wb') as f:
            pickle.dump(group_labels, f)
        with open(os.path.join(self.out_folder, phase + '_individual_label.pkl'), 'wb') as f:
            pickle.dump(individual_labels, f)

        # Save pose data
        res_skeleton = np.array(res_skeleton)
        np.save(os.path.join(self.out_folder, phase + '_data.npy'), res_skeleton)

        # Save ball data
        res_ball = np.array(res_ball)
        np.save(os.path.join(self.out_folder, phase + '_object_data.npy'), res_ball)

    def start(self):
        for phase in ['train', 'test']:
            print('Phase: {}'.format(phase))
            self.gendata(phase)