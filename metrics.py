import torch

@torch.no_grad()
def accuracy(pred, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    """
        pred: prediction logits in [B, action_categories]
        target: ground truth label in [B]
    """
    if target.numel() == 0:
        return [torch.zeros([], device=pred.device)]
    maxk = max(topk)

    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(k, -1).float().sum(0).mul_(100.0)
        res.append(correct_k)
    return res


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def k_segment_iou(target_segments, candidate_segments):
    return np.stack(
        [segment_iou(target_segment, candidate_segments) \
            for target_segment in target_segments]
    )


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(np.float16) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index'].astype(np.int16)] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index'].astype(np.int16)] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float16)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float16)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    prediction_gbvn = prediction.groupby('video-id')

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred['score'].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[:max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(this_pred[['t-start', 't-end']].values[top_kx_idx],
                                 this_gt[['t-start', 't-end']].values)

        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[:k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall

class LocEval(object):
    """Adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py"""

    def __init__(
        self,
        ground_truth,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_k=(1, 5),
        num_workers=8
    ):

        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.ap = None
        self.num_workers = num_workers

        self.ground_truth = pd.DataFrame({
            'video-id' : ground_truth['video-id'],
            't-start' : ground_truth['t-start'],
            't-end': ground_truth['t-end'],
            'label': ground_truth['label']
        })

        # remove labels that does not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def wrapper_compute_topkx_recall(self, preds):
        """Computes Top-kx recall for each class in the subset.
        """
        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_topkx_recall_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            recall[...,cidx] = results[i]

        return recall

    def evaluate(self, preds):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be a python dict item with numpy arrays as the values
        """

        # move dict to pd dataframe
        preds = pd.DataFrame({
            'video-id' : preds['video-id'],
            't-start' : preds['t-start'],
            't-end': preds['t-end'],
            'label': preds['label'],
            'score': preds['score']
        })
        # always reset ap
        self.ap = None

        # compute mAP
        self.ap = self.wrapper_compute_average_precision(preds)
        self.recall = self.wrapper_compute_topkx_recall(preds)
        mAP = self.ap.mean(axis=1)
        mRecall = self.recall.mean(axis=2)
        average_mAP = mAP.mean()

        info_block = ''
        for tiou, tiou_mAP, tiou_mRecall in zip(self.tiou_thresholds, mAP, mRecall):
            info_block += '\n|tIoU = {:.2f}: '.format(tiou)
            info_block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
            for idx, k in enumerate(self.top_k):
                info_block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
        info_block += '\n Average: mAP = {:>4.2f} (%)'.format(average_mAP*100)

        return mAP, average_mAP, mRecall, info_block