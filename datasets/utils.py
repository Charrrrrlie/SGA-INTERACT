import torch
import random
import numpy as np

from datasets.constants import COCO_KEYPOINT_INDEXES, OKS_sigmas
from datasets.constants import COCO_KEYPOINT_HORIZONTAL_FLIPPED, KEYPOINT_PURTURB_RANGE, KEYPOINT3D_PURTRUB_RANGE

def get_djoints(joints):
    """
        calculate the derivative of joints
    """
    d_joints = np.zeros_like(joints)
    d_joints[1:] = joints[1:] - joints[:-1]
    return d_joints

def personwise_normalization(points,
                             offset_points=None, 
                             scale_distance_point_pairs=None,
                             scale_distance_fn=None,
                             scale_unit=None):
    '''
    https://github.com/google-research/google-research/blob/master/poem/core/keypoint_profiles.py#L483

    Args:
        points: A tensor for points. Shape = [..., num_points, point_dim].
        offset_points: A list of points to compute the center point. 
                If a single point is specified, the point will be
                used as the center point. If multiple points are specified, the centers of their
                corresponding coordinates will be used as the center point.
        scale_distance_point_pairs: A list of tuples and each tuple is a pair of point lists. 
                The points will be converted into point indices later. Then, for
                example, use [([0], [1]), ([2, 3], [4])] to compute scale distance as the
                scale_distance_fn of the distance between point_0 and point_1 and the distance
                between the center(point_2, point_3) and point_4.
        scale_distance_fn: A  function handle for distance, e.g., torch.sum.
        scale_unit: A scalar for the scale unit whose value the scale distance will
                be scaled to.
    '''
    if not offset_points:
        offset_points = ['left_hip', 'right_hip']
    if not scale_distance_point_pairs:
        scale_distance_point_pairs = [(['left_shoulder'], ['right_shoulder']),
                                        (['left_shoulder'], ['left_hip']),
                                        (['left_shoulder'], ['right_hip']),
                                        (['right_shoulder'], ['left_hip']),
                                        (['right_shoulder'], ['right_hip']),
                                        (['left_hip'], ['right_hip'])]
    if not scale_distance_fn:
        scale_distance_fn = 'max'  # {sum, max, mean}
    if not scale_unit:
        scale_unit = 1.0

    # https://github.com/google-research/google-research/blob/master/poem/core/keypoint_profiles.py#L979

    # get indices instead of keypoint names
    offset_point_indices = [COCO_KEYPOINT_INDEXES[keypoint] for keypoint in offset_points]
    scale_distance_point_index_pairs = []
    for (points_i, points_j) in scale_distance_point_pairs:
        points_i_indices = []
        for keypoint in points_i:
            points_i_indices.append(COCO_KEYPOINT_INDEXES[keypoint])
            
        points_j_indices = []
        for keypoint in points_j:
            points_j_indices.append(COCO_KEYPOINT_INDEXES[keypoint])
        
        scale_distance_point_index_pairs.append((points_i_indices, points_j_indices))

    def get_points(points, indices):
        """Gets points as the centers of points at specified indices.

            Args:
            points: A tensor for points. Shape = [..., num_points, point_dim].
            indices: A list of integers for point indices.

            Returns:
            A tensor for (center) points. Shape = [..., 1, point_dim].

            Raises:
            ValueError: If `indices` is empty.
        """
        if not indices:
            raise ValueError('`Indices` must be non-empty.')
        points = points[:, indices, :]
        if len(indices) == 1:
            return points
        return torch.mean(points, dim=-2, keepdim=True)

    offset_point = get_points(points, offset_point_indices)

    def compute_l2_distances(lhs, rhs, squared=False, keepdim=False):
        """Computes (optionally squared) L2 distances between points.

            Args:
            lhs: A tensor for LHS points. Shape = [..., point_dim].
            rhs: A tensor for RHS points. Shape = [..., point_dim].
            squared: A boolean for whether to compute squared L2 distance instead.
            keepdims: A boolean for whether to keep the reduced `point_dim` dimension
                (of length 1) in the result distance tensor.

            Returns:
            A tensor for L2 distances. Shape = [..., 1] if `keepdims` is True, or [...]
                otherwise.
        """
        squared_l2_distances = torch.sum(
            (lhs - rhs)**2, dim=-1, keepdim=keepdim)
        return torch.sqrt(squared_l2_distances) if squared else squared_l2_distances

    def compute_scale_distances():
        sub_scale_distances_list = []
        for lhs_indices, rhs_indices in scale_distance_point_index_pairs:
            lhs_points = get_points(points, lhs_indices)  # Shape = [..., 1, point_dim]
            rhs_points = get_points(points, rhs_indices)  # Shape = [..., 1, point_dim]
            sub_scale_distances_list.append(
                compute_l2_distances(lhs_points, rhs_points, squared=True, keepdim=True))  # Euclidean distance

        sub_scale_distances = torch.cat(sub_scale_distances_list, dim=-1)

        if scale_distance_fn == 'sum':
            return torch.sum(sub_scale_distances, dim=-1, keepdim=True)
        elif scale_distance_fn == 'max':
            return torch.max(sub_scale_distances, dim=-1, keepdim=True).values
        elif scale_distance_fn == 'mean':
            return torch.mean(sub_scale_distances, dim=-1, keepdim=True)
        else:
            raise ValueError('Please check whether scale_distance_fn is supported!')

    scale_distances = compute_scale_distances()

    normalized_points = (points - offset_point) / (scale_distances * scale_unit + 1e-12)  # in case divide by 0
    return normalized_points


def oks_one_keypoint_compute(keypoint, keypoint_prev, box, box_prev, require_norm=False, OW=720.0, OH=1280.0):
    # Arguments:
    # - keypoint: (3,) - x, y, type 
    # - keypoint_prev: (3,) - x, y, type, this keypoint in previous frame
    # - box: (4,) bounding box of the person
    # - box_previous: (4,), this bounding box in previous frame
    
    keypoint_type = keypoint[2]
    
    if require_norm:
        keypoint[0] /= OH
        keypoint[1] /= OW
        keypoint_prev[0] /= OH
        keypoint_prev[1] /= OW
    
    y1,x1,y2,x2 = box
    if require_norm:
        box = [x1, y1, x2, y2] 
    else:
        box = [x1*OH, y1*OW, x2*OH, y2*OW] 
    area = (box[2]-box[0]) * (box[3]-box[1])
    
    y1,x1,y2,x2 = box_prev
    if require_norm:
        box_prev = [x1, y1, x2, y2] 
    else:
        box_prev = [x1*OH, y1*OW, x2*OH, y2*OW] 
    area_prev = (box_prev[2]-box_prev[0]) * (box_prev[3]-box_prev[1])
    
    avg_area = (area + area_prev) / 2.0
    if avg_area == 0:  # happen when box and keypoint coords are just 0s
        return 0.0
    
    dist = np.linalg.norm(keypoint[:2]-keypoint_prev[:2])

    oks = np.exp(- dist**2 / ( 2 * avg_area * OKS_sigmas[int(keypoint_type)]**2))
    
    return oks


def oks_keypoint_3d_compute(keypoints):
    # Arguments:
    # - keypoints: (T, N, J, 3) - x, y, z
    # - keypoints_prev: (T, N, J, 3) - x, y, z, this keypoint in previous frame

    def get_box_from_kp(kp):

        x = kp[..., 0]
        y = kp[..., 1]
        z = kp[..., 2]
        box = np.zeros((kp.shape[0], kp.shape[1], 6))
        box[..., 0] = np.min(x, axis=-1) - 0.1
        box[..., 1] = np.min(y, axis=-1) - 0.1
        box[..., 2] = np.min(z, axis=-1) - 0.1
        box[..., 3] = np.max(x, axis=-1) + 0.1
        box[..., 4] = np.max(y, axis=-1) + 0.1
        box[..., 5] = np.max(z, axis=-1) + 0.1

        return box

    T, N, J, C = keypoints.shape
    oks = np.zeros((T, N, J))
    oks[0, ...] = 0.0

    prev_keypoints = keypoints[0:-1, ...]
    cur_keypoints = keypoints[1:, ...]

    prev_box3d = get_box_from_kp(prev_keypoints)
    cur_box3d = get_box_from_kp(cur_keypoints)

    prev_box_area = (prev_box3d[..., 3] - prev_box3d[..., 0]) * (prev_box3d[..., 4] - prev_box3d[..., 1]) * (prev_box3d[..., 5] - prev_box3d[..., 2])
    cur_box_area = (cur_box3d[..., 3] - cur_box3d[..., 0]) * (cur_box3d[..., 4] - cur_box3d[..., 1]) * (cur_box3d[..., 5] - cur_box3d[..., 2])

    avg_area = (prev_box_area + cur_box_area) / 2.0
    avg_area = avg_area.reshape((T - 1, N, 1))

    dist = np.linalg.norm(cur_keypoints - prev_keypoints, axis=-1)
    oks[1:, ...] = np.exp(-dist**2 / (1e-6 + 2 * avg_area * OKS_sigmas.reshape((1, 1, -1)) **2))
    oks = np.clip(oks, 0, 1)
    return oks.reshape((T, N, J, 1))

# ----------------- MPGCN -----------------
def graph_processing(data, graph_form, processing):
    C, T, V, M = data.shape
    num_person = 1 if len(graph_form.split('-')) == 1 else int(graph_form.split('-')[1])

    if num_person > 1:
        if processing == 'default':
            multi_person_data = np.zeros([C, T, V*M, 1])
            for i in range(M):
                multi_person_data[:, :, V*i:V*i+V, 0] = data[:, :, :, i]
        elif processing == 'two-group':
            multi_person_data = np.zeros([C, T, V*M//2, 2])
            for i in range(M//2):
                multi_person_data[:, :, V*i:V*i+V, 0] = data[:, :, :, i]
                multi_person_data[:, :, V*i:V*i+V, 1] = data[:, :, :, i+M//2]
        else:
            raise ValueError('Error: Wrong in loading processing configs')
        return multi_person_data
    return data

def multi_input(data, conn, inputs, centers):
    C, T, V, M = data.shape

    joint_motion = np.zeros((C*2, T, V, M))
    bone = np.zeros((C*2, T, V, M))
    bone_motion = np.zeros((C*2, T, V, M))

    centers_ori = np.array(centers).copy()
    centers = np.clip(centers, 0, V-1)
    joint = np.concatenate([data,  data - data[:, :, centers, :]], axis=0)
    joint[C:, :, np.array(centers_ori) < 0, :] = 0

    joint_motion[:C, :-2, :, :] = data[:, 1:-1, :, :] - data[:, :-2, :, :]
    joint_motion[C:, :-2, :, :] = data[:, 2:, :, :] - data[:, :-2, :, :]

    conn_ori = np.array(conn).copy()
    conn = np.clip(conn, 0, V-1)
    bone[:C] = data - data[:, :, conn, :]
    bone[:C, :, np.array(conn_ori) < 0, :] = 0

    bone_length = np.sum(bone[:C, :, :, :] ** 2, axis=0)
    bone_length = np.sqrt(bone_length) + 0.0001
    bone[C:, :, :, :] = np.arccos(bone[:C, :, :, :] / bone_length)

    bone_motion[:C, :-2, :, :] = bone[:C, 1:-1, :, :] - bone[:C, :-2, :, :]
    bone_motion[C:, :-2, :, :] = bone[C:, 1:-1, :, :] - bone[C:, :-2, :, :]

    data_new = []
    if inputs.isupper():
        if 'J' in inputs:
            data_new.append(joint)
        if 'V' in inputs:
            data_new.append(joint_motion)
        if 'B' in inputs:
            data_new.append(bone)
        if 'M' in inputs:
            data_new.append(bone_motion)
    else:
        raise ValueError('Error: No input feature!')
    return np.stack(data_new, axis=0)

# ----------------- STGCN -----------------
def COCO_to_OpenPose25(joints):
    """
    Convert COCO keypoints to OpenPose25 keypoints.
    Args:
        joints: (T, N, J, 3), J=17 for COCO
    """
    T, N, _, _ = joints.shape
    joints_openpose = np.zeros((T, N, 25, 3))
    mapping_idx = np.array([
        0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11
    ])
    joints_openpose[..., mapping_idx, :] = joints

    joints_openpose[..., 8, :] = (joints_openpose[..., 11, :] + joints_openpose[..., 12, :]) / 2.0

    return joints_openpose

# ----------------- Augmentation -----------------
def horizontal_flip_augment_joint(joint_raw, frames, horizontal_flip_augment_joint_randomness, image_w, add_purturbation=False, randomness_set=False, index=0):
    for t in frames:
        for n in range(len(joint_raw[t])):
            if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                continue
            for j in range(len(joint_raw[t][n])):
                joint_raw[t][n, j, 0] = image_w - joint_raw[t][n, j, 0]  # flip joint coordinates
                if add_purturbation:
                    if not randomness_set:
                        horizontal_flip_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                            -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                    joint_raw[t][n, j, 0] += horizontal_flip_augment_joint_randomness[index][(t, n, j)]
                joint_raw[t][n, j, 2] = COCO_KEYPOINT_HORIZONTAL_FLIPPED[joint_raw[t][n, j, 2]]  # joint class type has to be flipped
            joint_raw[t][n] = joint_raw[t][n][joint_raw[t][n][:, 2].argsort()]  # sort by joint type class id
    return joint_raw


def horizontal_move_augment_joint(joint_raw, frames, horizontal_move_augment_joint_randomness, add_purturbation=False, randomness_set=True, index=0, max_horizontal_diff=10.0, ball_trajectory=None):
        horizontal_change = np.random.uniform(low=-max_horizontal_diff, high=max_horizontal_diff)
        for t in frames:
            for n in range(len(joint_raw[t])):
                if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                    continue
                for j in range(len(joint_raw[t][n])):
                    joint_raw[t][n, j, 0] += horizontal_change  # horizontally move joint
                    if add_purturbation:
                        if not randomness_set:
                            horizontal_move_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                                -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                        joint_raw[t][n, j, 0] += horizontal_move_augment_joint_randomness[index][(t, n, j)]
        if ball_trajectory is not None:
            for t in range(len(ball_trajectory)):
                 ball_trajectory[t, 0] += horizontal_change
            return joint_raw, ball_trajectory
        else:
            return joint_raw

def vertical_move_augment_joint(joint_raw, frames, vertical_move_augment_joint_randomness, add_purturbation=False, randomness_set=True, index=0, max_vertical_diff=10.0, ball_trajectory=None):
    vertical_change = np.random.uniform(low=-max_vertical_diff, high=max_vertical_diff)
    for t in frames:
        for n in range(len(joint_raw[t])):
            if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                continue
            for j in range(len(joint_raw[t][n])):
                joint_raw[t][n, j, 1] += vertical_change  # vertically move joint
                if add_purturbation:
                    if not randomness_set:
                        vertical_move_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                            -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                    joint_raw[t][n, j, 1] += vertical_move_augment_joint_randomness[index][(t, n, j)]
    if ball_trajectory is not None:
        for t in range(len(ball_trajectory)):
                ball_trajectory[t, 1] += vertical_change
        return joint_raw, ball_trajectory
    else:
        return joint_raw

def horizontal_flip_ball_trajectory(ball_trajectory, image_w):
    # ball_trajectory: (T, 2)
    for t in range(len(ball_trajectory)):
            ball_trajectory[t, 0] = image_w - ball_trajectory[t, 0]
    return ball_trajectory

def agent_dropout_augment_joint(joint_feats, frames, agent_dropout_augment_randomness):
    # joint_feats: (N, J, T, d)
    chosen_frame = agent_dropout_augment_randomness[0]
    chosen_person = agent_dropout_augment_randomness[1]
    feature_dim = joint_feats.shape[3]

    joint_feats[chosen_person, :, frames.index(chosen_frame), :] = 0.0
    return joint_feats


# ----------------- Augmentation 3D -----------------
def flip_augment_joint3D(joints, image_w, axis, add_purturbation=False):
    """
    Flip player in given axis. Note that we keep (J, 3) integrity. Recommand axis=1 for global Y-axis flip.
    Args:
        joints: (T, N, J, 3)
        image_w: int
        axis: int
    """
    joints[:, :, :, axis] = image_w - joints[:, :, :, axis]
    if add_purturbation:
        # get random shape like joints
        randomness = np.random.uniform(-KEYPOINT3D_PURTRUB_RANGE, KEYPOINT3D_PURTRUB_RANGE, size=joints.shape[:-2])
        randomness = randomness[..., np.newaxis]
        joints[:, :, :, axis] += randomness

    return joints


def move_augment_joint3D(joints, axis, max_shift=0.15, add_purturbation=False):
    """
    Random move player in given axis. Note that we keep (J, 3) integrity.
    Args:
        joints: (T, N, J, 3)
        axis: int
    """
    change_val = np.random.uniform(low=-max_shift, high=max_shift, size=joints.shape[:-2])
    change_val = change_val[..., np.newaxis]
    joints[:, :, :, axis] += change_val
    if add_purturbation:
        # get random shape like joints
        randomness = np.random.uniform(-KEYPOINT3D_PURTRUB_RANGE, KEYPOINT3D_PURTRUB_RANGE, size=joints.shape[:-2])
        randomness = randomness[..., np.newaxis]
        joints[:, :, :, axis] += randomness

    return joints

def agent_dropout_augment_joint3D(joints, downsample_ratio):
    """
    Randomly drop one player in random frames. Note that we use zeros to represent the dropped player.
    Args:
        joints: (T, N, J, 3),
        downsample_ratio: float
    """
    assert downsample_ratio < 1.0
    frame_num = int(joints.shape[0] * downsample_ratio)

    chosen_frame = np.random.choice(joints.shape[0], size=frame_num, replace=False)
    chosen_person = np.random.randint(joints.shape[1])

    joints[chosen_frame, chosen_person, :, :] = 0.0

    return joints

def agent_temporal_augment3D(joints, downsample_ratio):
    """
    Temporally downsample the joints. It changes velocity of motions.
    Args:
        joints: (T, N, J, 3)
        downsample_ratio: float
    """
    assert downsample_ratio < 1.0

    source_T = joints.shape[0]
    target_T = int(source_T * downsample_ratio)

    chosen_frame = np.random.choice(source_T, target_T, replace=False)
    joints = joints[chosen_frame]

    return joints, chosen_frame