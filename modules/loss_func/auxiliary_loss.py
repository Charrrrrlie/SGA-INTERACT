import torch
import torch.nn.functional as F

def sinkhorn(scores, epsilon=0.05, nmb_iters=3):
    with torch.no_grad():
        Q = torch.exp(scores / epsilon).t() 
        K, B = Q.shape
    
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        u = torch.zeros(K).cuda()
        r = torch.ones(K).cuda() / K
        c = torch.ones(B).cuda() / B

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)

            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def swap_prediction(p_t, p_s, q_t, q_s):
    loss = - 0.5 * (
        torch.mean(
            torch.sum(
                q_t * F.log_softmax(p_s, dim=1), dim=1)
        ) + torch.mean(torch.sum(q_s * F.log_softmax(p_t, dim=1), dim=1)))
    return loss

def compute_contrastive_loss(scores, loss_weight, sinkhorn_iterations, temperature):
    # learning the cluster assignment and computing the loss
    scores_f = scores[0]
    scores_m = scores[1]
    scores_c = scores[2]
    scores_g = scores[3]

    # compute assignments
    with torch.no_grad(): 
        q_f = sinkhorn(scores_f, nmb_iters=sinkhorn_iterations)
        q_m = sinkhorn(scores_m, nmb_iters=sinkhorn_iterations)
        q_c = sinkhorn(scores_c, nmb_iters=sinkhorn_iterations)
        q_g = sinkhorn(scores_g, nmb_iters=sinkhorn_iterations)

    # swap prediction problem
    p_f = scores_f / temperature
    p_m = scores_m / temperature
    p_c = scores_c / temperature
    p_g = scores_g / temperature

    contrastive_clustering_loss = loss_weight * (
        swap_prediction(p_f, p_m, q_f, q_m) + 
        swap_prediction(p_f, p_c, q_f, q_c) +
        swap_prediction(p_f, p_g, q_f, q_g) +
        swap_prediction(p_m, p_c, q_m, q_c) +
        swap_prediction(p_m, p_g, q_m, q_g) +
        swap_prediction(p_c, p_g, q_c, q_g)
    ) / 6.0  # 6 pairs of views

    return contrastive_clustering_loss