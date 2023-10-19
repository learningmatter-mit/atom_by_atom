import numpy as np
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
)
from torch.optim import SGD, Adam, Adadelta, AdamW#, NAdam, RAdam
import torch
from persite_painn.train.loss import mse_operation
from persite_painn.train.metric import mae_operation


def get_optimizer(
    optim,
    trainable_params,
    lr,
    weight_decay,
):
    if optim == "SGD":
        print("SGD Optimizer")
        optimizer = SGD(
            trainable_params,
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay,
        )
    elif optim == "Adam":
        print("Adam Optimizer")
        optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "Nadam":
        print("NAdam Optimizer")
        optimizer = NAdam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "AdamW":
        print("AdamW Optimizer")
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "Adadelta":
        print("Adadelta Optimizer")
        optimizer = Adadelta(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "Radam":
        print("RAdam Optimizer")
        optimizer = RAdam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise NameError("Optimizer not implemented --optim")

    return optimizer


def get_scheduler(sched, optimizer, epochs, lr_update_rate=30, lr_milestones=[100]):
    if sched == "cos_anneal":
        print("Cosine anneal scheduler")
        scheduler = CosineAnnealingLR(optimizer, lr_update_rate)
    elif sched == "cos_anneal_warm_restart":
        print("Cosine anneal with warm restarts scheduler")
        scheduler = CosineAnnealingWarmRestarts(optimizer, lr_update_rate)
    elif sched == "reduce_on_plateau":
        print("Reduce on plateau scheduler")
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.5,
            threshold=0.01,
            verbose=True,
            threshold_mode="abs",
            patience=15,
        )
    elif sched == "multi_step":
        print("Multi-step scheduler")
        lr_milestones = np.arange(
            lr_update_rate, epochs + lr_update_rate, lr_update_rate
        )
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    else:
        raise NameError("Choose within cos_anneal, reduce_on_plateau, multi_stp")
    return scheduler


def build_loss_metric_fn(
    loss_coef,
    operation,
    correspondence_keys=None,
    normalizer=None,
):
    """
    Build a general  loss function.
    Args:
        loss_coef (dict): dictionary containing the weight coefficients
            for each property being predicted.
            Example: `loss_coef = {'energy': rho, 'force': 1}`
        operation (function): a function that acts on the prediction and
            the target to produce a result (e.g. square it, put it through
            cross-entropy, etc.)
        correspondence_keys (dict): a dictionary that links an output key to
            a different key in the dataset.
            Example: correspondence_keys = {"autopology_energy_grad": "energy_grad"}
            This tells us that if we see "autopology_energy_grad" show up in the
            loss coefficient, then the loss should be calculated between
            the network's output "autopology_energy_grad" and the data in the dataset
            given by "energy_grad". This is useful if we're only outputting one quantity,
            such as the energy gradient, but we want two different outputs (such as
            "energy_grad" and "autopology_energy_grad") to be compared to it.
    Returns:
        mean squared error loss function
    """

    correspondence_keys = {} if (correspondence_keys is None) else correspondence_keys

    def loss_fn(results, ground_truth):
        """Calculates the MSE between ground_truth and results.
        Args:
            results (dict):  e.g. `{'energy': 4, 'force': [1, 2, 2]}`
            ground_truth (dict): e.g. `{'energy': 2, 'force': [0, 0, 0]}`
        Returns:
            loss (torch.Tensor)
        """

        # assert all([k in results.keys() for k in loss_coef.keys()])
        # assert all([k in [*ground_truth.keys(), *correspondence_keys.keys()]
        #             for k in loss_coef.keys()])

        loss = 0.0
        for key, coef in loss_coef.items():

            if key not in ground_truth.keys():
                ground_key = correspondence_keys[key]
            else:
                ground_key = key

            if normalizer is not None:
                targ = normalizer[ground_key].norm(ground_truth[ground_key])
            else:
                targ = ground_truth[ground_key]

            pred = results[key].view(targ.shape)

            # select only properties which are given
            valid_idx = torch.bitwise_not(torch.isnan(targ))

            targ = targ[valid_idx]
            pred = pred[valid_idx]

            if len(targ) != 0:
                diff = operation(prediction=pred, target=targ)
                err_sq = coef * torch.mean(diff)
                loss += err_sq

        return loss

    return loss_fn


def get_loss_metric_fn(
    loss_coeff,
    correspondence_keys,
    operation_name,
    normalizer=None,
):
    if operation_name == "MSE":
        loss_fn = build_loss_metric_fn(
            loss_coef=loss_coeff,
            operation=mse_operation,
            correspondence_keys=correspondence_keys,
            normalizer=normalizer,
        )
    elif operation_name == "MAE":
        loss_fn = build_loss_metric_fn(
            loss_coef=loss_coeff,
            operation=mae_operation,
            correspondence_keys=correspondence_keys,
            normalizer=normalizer,
        )

    return loss_fn
