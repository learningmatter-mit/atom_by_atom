import torch
from .loss import sid_operation
from persite_painn.utils.tools import gaussian_smoothing


def sis_operation(prediction, target, sigma=2):

    filtered_target = gaussian_smoothing(target, sigma)
    filtered_pred = gaussian_smoothing(prediction, sigma)
    if filtered_target.dim() == 1:
        filtered_target = filtered_target.unsqueeze(0)
        filtered_pred = filtered_pred.unsqueeze(0)
    sid = sid_operation(filtered_pred, filtered_target)
    sis = 1 / (1 + sid)

    return sis


def sis_loss(prediction, target):
    loss = torch.mean(sis_operation(prediction=prediction, target_spectra=target))
    return loss


def mae_loss(
    prediction,
    target,
):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    flattened_pred = prediction.view(1, -1)
    flattened_targ = target.view(1, -1)
    assert flattened_pred.shape[0] == flattened_targ.shape[0]
    nan_mask = torch.isnan(flattened_targ)
    nan_mask = nan_mask.to(target.device)

    return torch.mean(torch.abs(flattened_pred[~nan_mask] - flattened_targ[~nan_mask]))


# def mae_operation(
#     prediction,
#     target,
# ):
#     """
#     Computes the mean absolute error between prediction and target

#     Parameters
#     ----------

#     prediction: torch.Tensor (N, 1)
#     target: torch.Tensor (N, 1)
#     """
#     flattened_pred = prediction.view(1, -1)
#     flattened_targ = target.view(1, -1)
#     assert flattened_pred.shape[0] == flattened_targ.shape[0]

#     return torch.abs(flattened_pred - flattened_targ)


def mae_operation(prediction, target):
    targ = target.to(torch.float)
    diff = abs(targ - prediction)
    return diff
