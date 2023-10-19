import torch


def sid_operation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    # normalize the model spectra before comparison
    nan_mask = torch.isnan(target) + torch.isnan(prediction)
    nan_mask = nan_mask.to(target.device)
    zero_sub = torch.zeros_like(target, device=target.device)
    one_sub = torch.ones_like(prediction, device=target.device)
    sum_model_spectra = torch.sum(
        torch.where(nan_mask, zero_sub, prediction), dim=1, keepdim=True
    )

    model_spectra = torch.div(prediction, sum_model_spectra)
    model_spectra = torch.where(nan_mask, one_sub, model_spectra) + eps
    target_spectra = torch.where(nan_mask, one_sub, target) + eps

    sid = torch.sum(
        torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra)
        + torch.mul(
            torch.log(torch.div(target_spectra, model_spectra)), target_spectra
        ),
        dim=1,
        keepdim=True,
    )

    return sid


def sid_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
):

    loss = torch.mean(sid_operation(prediction=prediction, target=target))
    return loss


def stmse_operation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    # normalize the model spectra before comparison
    nan_mask = torch.isnan(target) + torch.isnan(prediction)
    nan_mask = nan_mask.to(target.device)
    zero_sub = torch.zeros_like(target, device=target.device)
    one_sub = torch.ones_like(prediction, device=target.device)

    sum_model_spectra = torch.sum(
        torch.where(nan_mask, zero_sub, prediction), dim=1, keepdim=True
    )
    model_spectra = torch.div(prediction, sum_model_spectra)
    target_spectra = torch.where(nan_mask, one_sub, target) + eps
    model_spectra = torch.where(nan_mask, one_sub, model_spectra) + eps

    stmse = torch.mean(
        torch.div((model_spectra - target_spectra) ** 2, target_spectra),
        dim=1,
        keepdim=True,
    )

    return stmse


def stmse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
):
    loss = torch.mean(stmse_operation(prediction=prediction, target=target))

    return loss


def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    flattened_pred = prediction.view(1, -1)
    flattened_targ = target.view(1, -1)
    assert flattened_pred.shape[0] == flattened_targ.shape[0]
    nan_mask = torch.isnan(flattened_targ)
    nan_mask = nan_mask.to(target.device)

    loss = torch.mean((flattened_pred[~nan_mask] - flattened_targ[~nan_mask]) ** 2)

    return loss


# def mse_operation(
#     prediction: torch.Tensor,
#     target: torch.Tensor,
# ) -> torch.Tensor:
#     flattened_pred = prediction.view(1, -1)
#     flattened_targ = target.view(1, -1)
#     assert flattened_pred.shape[0] == flattened_targ.shape[0]
#     # nan_mask = torch.isnan(flattened_targ)
#     # nan_mask = nan_mask.to(target.device)
#     loss = (flattened_pred - flattened_targ) ** 2

#     return loss


def mse_operation(prediction, target):
    """
    Square the difference of target and predicted.
    Args:
        targ (torch.Tensor): target
        pred (torch.Tensor): prediction
    Returns:
        diff (torch.Tensor): difference squared
    """
    targ = target.to(torch.float)
    diff = (targ - prediction) ** 2
    return diff
