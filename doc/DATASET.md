
## SGA-INTERACT Structure
Download the dataset and put it in `data` folder. The structure of the dataset is as follows:
```
basketball
├── annots
│   ├── tactic
│   │   ├── <Clip_Name>_tactic.pkl # S*_** denotes different source videos
│   │   ├── ...
│   ├── ball
│   │   ├── <Clip_Name>_ball_traj.pkl
│   │   ├── ...
├── joints
│   ├── <Clip_Name>_pose.npy
│   ├── ...
├── GAL_test_split_0.3ratio_info.pkl # GAL test split
├── GAL_train_split_0.3ratio_info.pkl # GAL train split
├── GAR_test_split_0.3ratio_info.pkl # GAR test split
└── GAR_train_split_0.3ratio_info.pkl # GAR train split
```
#
**Tactic annotation** contains a dict of dict.
Note that the start_time and end_time are in seconds and can be converted to frames by frame rate (50 for all).
```
{
    'Action': {
        'Action_Name': [[start_time1, end_time1], [start_time2, end_time2], ...],
        ...
    },
    'Offensive': 'Blue' or 'White' # guest / host team
}
```

#
**Ball Possession annotation** contains a dict of list.

Note that each item denotes the period of time that the player is holding the ball.
```
{
    'Team_Playerid': [[start_time1, end_time1], [start_time2, end_time2], ...],
    ...
}
```
#
**Joint npy file** contains a dict of numpy array.

The joints are in a left-hand world coordinate system. Its origin lies at the left corner (facing basket) of the court and the x-axis is the short side of the court, the y-axis is the long side of the court.
```
{
    'Team_Playerid': np.array([num_frames, num_joints, 3]),
}
```

You can get the same scene visualization of our paper by `demo.py`.