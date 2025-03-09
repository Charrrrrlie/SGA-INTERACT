import numpy as np

BASKETBALL_ACTION_SET = {
    'Basket Cut',
    'Cross',
    'DHO',
    'HO',
    'FHO',
    'Keep',
    'Playmaking',
    'Pin',
    'Flare',
    'Reverse',
    'Isolation',
    'PnR',
    'Slip',
    'Post',
    'Pop',
    'Up',
    'Down',
    'Exit',
    'Through',
    'Shuffle',
    'Baseline Cut',
}

VOLLEYBALL_ACTION_DICT = {
    0: {'r_set', 'r-set'},
    1: {'l_set', 'l-set'},
    2: {'r_spike', 'r-spike'},
    3: {'l_spike', 'l-spike'},
    4: {'r_pass', 'r-pass'},
    5: {'l_pass', 'l-pass'},
    6: {'r_winpoint', 'r-winpoint'},
    7: {'l_winpoint', 'l-winpoint'}
}

COCO_KEYPOINT_INDEXES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

COCO_KEYPOINT_HORIZONTAL_FLIPPED = {
    0: 0,
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: 6,
    6: 5,
    7: 8,
    8: 7,
    9: 10,
    10: 9,
    11: 12,
    12: 11,
    13: 14,
    14: 13,
    15: 16,
    16: 15
}

KEYPOINT_PURTURB_RANGE = 1.0
KEYPOINT3D_PURTRUB_RANGE = 0.1

OKS_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0