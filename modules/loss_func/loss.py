import torch

def compute_recognition_loss(loss_args, pred_logits, labels):
    """
        loss_args: loss arguments
        pred_logits: predicted logits in [B, activity_categories]
        labels: ground truth labels in [B]
    """
    if isinstance(pred_logits, list):
        device = pred_logits[0].device
    else:
        device = pred_logits.device
        pred_logits = [pred_logits]
    criterion = torch.nn.CrossEntropyLoss().to(device)

    loss = 0
    for i, pred in enumerate(pred_logits):
        if i == len(pred_logits) - 1 and i != 0:
            loss += criterion(pred, labels) * loss_args.loss_coe_last
        else:
            loss += criterion(pred, labels) * loss_args.loss_coe

    return loss

# --- COMPOSER ---
def compute_multi_weighted_recognition_loss(loss_args, group_pred, group_label, person_pred=None, person_labels=None):
    """
        loss_args: loss arguments
        group_pred: list of list of group prediction logits in num_TNT_layers * 4 * [B, activity_categories]
        person_pred: list of person prediction logits in [B * N, action_categories]
        group_label: ground truth group label in [B]
        person_labels: ground truth person label in [B * N]
    """
    loss_group = 0.0
    loss_person = 0.0
    if loss_args.use_group_activity_weights:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(loss_args.group_activity_weights))
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if 'use_person_action_weights' in loss_args and loss_args.use_person_action_weights:
        criterion_person = torch.nn.CrossEntropyLoss(weight=torch.Tensor(loss_args.person_action_weights))
    else:
        criterion_person = torch.nn.CrossEntropyLoss()

    criterion = criterion.to(group_label.device)
    criterion_person = criterion_person.to(group_label.device)

    for l in range(len(group_pred)):
        if l == len(group_pred) - 1:  # if last layer
            loss_group += loss_args.loss_coe_last_TNT * (
                loss_args.loss_coe_fine * criterion(group_pred[l][0], group_label) + 
                loss_args.loss_coe_mid * criterion(group_pred[l][1], group_label) +
                loss_args.loss_coe_coarse * criterion(group_pred[l][2], group_label) +
                loss_args.loss_coe_group * criterion(group_pred[l][3], group_label))
            
            if person_pred is not None and person_labels is not None:
                loss_person += loss_args.loss_coe_last_TNT * loss_args.loss_coe_person * criterion_person(person_pred[l], person_labels)

        else:  # not last layer
            loss_group += (
                loss_args.loss_coe_fine * criterion(group_pred[l][0], group_label) + 
                loss_args.loss_coe_mid * criterion(group_pred[l][1], group_label) + 
                loss_args.loss_coe_coarse * criterion(group_pred[l][2], group_label) +
                loss_args.loss_coe_group * criterion(group_pred[l][3], group_label))
            
            if person_pred is not None and person_labels is not None:
                loss_person += loss_args.loss_coe_person * criterion_person(person_pred[l], person_labels)

    return loss_group, loss_person