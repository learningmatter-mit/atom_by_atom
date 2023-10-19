import numpy as np
import torch

from persite_painn.train import AverageMeter
from persite_painn.utils import batch_to, inference


def shrink_batch(batch):
    """
    Exclude certain keys from the batch that take up a lot of memory
    """

    bad_keys = ['nbr_list', 'kj_idx', 'ji_idx',
                'angle_list']
    new_batch = {key: val for key, val in batch.items()
                 if key not in bad_keys}

    return new_batch


def test_model(
    model,
    test_loader,
    metric_fn,
    device,
    normalizer=None,
    multifidelity=False,
):
    """
    test the model performances
    Args:
        model: Model,
        test_loader: DataLoader,
        metric_fn: metric function,
        device: "cpu" or "cuda",
    Return:
        Lists of prediction, targets, ids, and metric
    """
    model.to(device)
    model.eval()
    test_targets = []
    test_preds = []
    test_ids = []
    test_targets_fidelity = []
    test_preds_fidelity = []
    metrics = AverageMeter()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to(batch, device)
            target = batch["target"]
            # Compute output
            output = inference(model=model, data=batch, normalizer=normalizer, output_key="target", device=device)
            if device == "cpu":
                metric_output = model(batch, inference=True)
            else:
                metric_output = model(batch)
            # measure accuracy and record loss
            metric = metric_fn(metric_output, batch)

            metrics.update(metric.cpu().item(), target.size(0))

            # Rearrange the outputs
            test_pred = output.data.cpu()
            test_target = target.detach().cpu()

            batch_ids = []
            count = 0
            num_bin = []
            for i, val in enumerate(batch["num_atoms"].detach().cpu().numpy()):
                count += val
                num_bin.append(count)
                if i == 0:
                    change = list(np.arange(val))
                else:
                    adding_val = num_bin[i - 1]
                    change = list(np.arange(val) + adding_val)
                batch_ids.append(change)

            test_preds += [test_pred[i].tolist() for i in batch_ids]
            test_targets += [test_target[i].tolist() for i in batch_ids]

            if multifidelity:
                target_fidelity = batch["fidelity"].detach().cpu()
                # Compute output
                output_fidelity = inference(
                    model=model, data=batch, normalizer=normalizer, output_key="fidelity", device=device
                ).data.cpu()

                test_preds_fidelity += [output_fidelity[i].tolist() for i in batch_ids]
                test_targets_fidelity += [
                    target_fidelity[i].tolist() for i in batch_ids
                ]

            metric_out = metrics.avg
            if isinstance(batch["name"], list):
                test_ids += batch["name"]
            else:
                test_ids += batch["name"].detach().tolist()

    return (
        test_preds,
        test_targets,
        test_ids,
        metric_out,
        test_preds_fidelity,
        test_targets_fidelity,
    )
