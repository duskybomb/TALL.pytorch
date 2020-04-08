import torch


def compute_loss_reg(sim_reg_mat, offsets):
    sim_score_mat, p_reg_mat, l_reg_mat = sim_reg_mat

    # unit matrix with -2
    input_size = sim_reg_mat.size(1)

    I = torch.eye(input_size).cuda()
    I_2 = -2 * I
    ones_matrix = torch.ones(input_size, input_size).cuda()

    mask_mat = I_2 + ones_matrix  # 56,56

    #               | -1  1   1...   |
    #   mask_mat =  | 1  -1   1...   |
    #               | 1   1  -1 ...  |

    alpha = 1.0 / input_size
    lambda_regression = 0.01
    batch_para_mat = alpha * ones_matrix
    para_mat = I + batch_para_mat

    loss_mat = torch.log(ones_matrix + torch.exp(mask_mat * sim_score_mat))
    loss_mat = loss_mat * para_mat
    loss_align = loss_mat.mean()

    # regression loss
    l_reg_diag = torch.mm(l_reg_mat * I, torch.ones(input_size, 1).cuda())
    p_reg_diag = torch.mm(p_reg_mat * I, torch.ones(input_size, 1).cuda())

    offset_pred = torch.cat([p_reg_diag, l_reg_diag], 1)

    loss_reg = torch.abs(offset_pred - offsets).mean()  # L1 loss

    return lambda_regression * loss_reg + loss_align, loss_align, loss_reg
