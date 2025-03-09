import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def draw_confusion_matrix(pred, gt, label_names, save_dir):
    cm = confusion_matrix(gt, pred)
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm,
                annot=True,
                fmt=".2f",
                cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names,
                cbar=False)
    plt.savefig(os.path.join(save_dir, 'cm.png'), bbox_inches='tight')
    plt.close()

# reference:
#color_dict = {
#     'DIN': '#D88778',
#     'STAtt': '#C28AD9',
#     'AT': '#5EAD91',
# }

# face_color_dict = {
#     'DIN': 'orange',
#     'STAtt': 'purple',
#     'AT': 'green',
# }
def draw_mAP_curve(mAP, thresholds, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds,
             mAP,
             marker='v',
             markeredgecolor='white',
             markerfacecolor='orange',
             markersize=8,           
             linestyle='-',
             c='#D88778',
             lw=2)
    plt.xlabel("t-IoU Threshold")
    plt.ylabel("mAP")
    plt.ylim(0, 10.5)
    plt.yticks(np.arange(0, 11, 2))
    plt.grid(True, linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_facecolor('#f8f8f8')
    plt.savefig(os.path.join(save_dir, 'mAP.png'), bbox_inches='tight')
    plt.close()