import argparse
import os
import pickle as pkl
import sys
import numpy as np
import json

import torch
from torch.utils.data import DataLoader, RandomSampler

from pymatgen.core.structure import Structure

from persite_painn.data.sampler import ImbalancedDatasetSampler
from persite_painn.data import collate_dicts
from persite_painn.data.builder import build_dataset, split_train_validation_test
from persite_painn.nn.builder import get_model, load_params_from_path
from persite_painn.train.builder import get_optimizer, get_scheduler, get_loss_metric_fn
from persite_painn.train.trainer import Trainer
from persite_painn.train.evaluate import test_model
from persite_painn.utils.train_utils import Normalizer
from persite_painn.data.preprocess import convert_site_prop

parser = argparse.ArgumentParser(description="Per-site PaiNN")
parser.add_argument("--data", default="", type=str, help="path to raw data")
parser.add_argument("--data_cache", default="dataset_cache", type=str, help="cache where data is / will be stored")
parser.add_argument("--results_dir", default="./results", type=str, help="saving directory")
parser.add_argument("--workers", default=0, type=int, help="number of data loading workers")
parser.add_argument("--epochs", default=50, type=int, help="number of total epochs to run")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch_size", default=64, type=int, help="mini-batch size")
parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint")
parser.add_argument("--cuda", default=0, type=int, help="GPU setting")
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
parser.add_argument("--early_stop_val", default=12, type=int, help="early stopping condition for validation loss update count")
parser.add_argument("--early_stop_train", default=0.1, type=float, help="early stopping condition for train loss tolerance")
parser.add_argument("--seed", default=42, type=int, help="Seed of random initialization to control the experiment")
parser.add_argument("--loss_fn", default="MSE", type=str, help="loss function")
parser.add_argument("--metric_fn", default="MAE", type=str, help="metric function")
parser.add_argument("--optimizer", default="Adam", type=str, help="optimizer")
parser.add_argument("--scheduler", default="reduce_on_plateau", type=str, help="scheduler")
parser.add_argument("--lr", default=0.002, type=float, help="initial learning rate")
parser.add_argument("--weight_decay", default=0.0, type=float, help="learning rate decay")
parser.add_argument("--val_size", default=0.1, type=float, help="validation set size")
parser.add_argument("--test_size", default=0.1, type=float, help="test set size")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(args):

    set_seed(args.seed)

    # Load data
    if os.path.exists(args.data_cache):
        print("Cached dataset exists...")
        dataset = torch.load(args.data_cache)
        print(f"Number of Data: {len(dataset)}")
    else:
        try:
            # read data from json
            with open(args.data) as f:
                data = json.load(f)

        except ValueError:
            print("Path to data should be given --data")
        else:
            print("Start making dataset...")

            # convert data to the correct format
            samples = {}
            for key in data.keys():
                samples.update({key:Structure.from_dict(data[key])})
            data = samples

            # get site properties using the first sample
            site_prop = list(data[list(data.keys())[0]].site_properties.keys())

            new_data = convert_site_prop(data, site_prop)
            dataset = build_dataset(
                raw_data=new_data,
                cutoff=4.0,
                multifidelity="False",
                seed=args.seed,
            )

            print(f"Number of Data: {len(dataset)}")
            print("Done creating dataset, caching...")
            dataset.save(args.data_cache)
            print("Done caching dataset")

    train_set, val_set, test_set = split_train_validation_test(
        dataset,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Normalizer
    normalizer = {}
    targs = []
    for batch in train_set:
        targs.append(batch["target"])

    targs = torch.concat(targs)
    normalizer_target = Normalizer(targs, "target")
    normalizer["target"] = normalizer_target
    modelparams.update({"means": {"target": normalizer_target.mean}})
    modelparams.update({"stddevs": {"target": normalizer_target.std}})

    # Get model
    model = get_model(
        modelparams,
        model_type=model_type,
        multifidelity="false",
    )
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Set optimizer
    optimizer = get_optimizer(
        optim=details["optim"],
        trainable_params=trainable_params,
        lr=details["lr"],
        weight_decay=details["weight_decay"],
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.start_epoch != 0:
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"]
                normalizer.load_state_dict(checkpoint["normalizer"])
            elif args.start_epoch == 0:
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                normalizer.load_state_dict(checkpoint["normalizer"])

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_metric = 1e10
        best_loss = 1e10

    # Set loss function
    if details["multifidelity"]:
        loss_coeff = {"fidelity": 1.0 - modelparams["loss_coeff"]["target"], "target": modelparams["loss_coeff"]["target"]}
        correspondence_keys = {"fidelity": "fidelity", "target": "target"}
    else:
        loss_coeff = {"target": 1.0}
        correspondence_keys = {"target": "target"}
    # Set loss function
    # TODO: normalier issue
    loss_fn = get_loss_metric_fn(
        loss_coeff=loss_coeff,
        correspondence_keys=correspondence_keys,
        operation_name=details["loss_fn"],
        normalizer=None,
    )
    # Set metric function
    metric_fn = get_loss_metric_fn(
        loss_coeff=loss_coeff,
        correspondence_keys=correspondence_keys,
        operation_name=details["metric_fn"],
        normalizer=None,
    )

    # Set scheduler
    scheduler = get_scheduler(
        sched=details["sched"], optimizer=optimizer, epochs=args.epochs
    )

    # Set DataLoader
    if details["multifidelity"]:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_dicts,
            sampler=ImbalancedDatasetSampler("classification", train_set.props),
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_dicts,
            sampler=RandomSampler(train_set),
        )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
    )
    # Save ids
    train_ids = []
    for item in train_set:
        if type(item["name"]) == str:
            train_ids.append(item["name"])
        else:
            train_ids.append(item["name"].item())
    val_ids = []
    for item in val_set:
        if type(item["name"]) == str:
            val_ids.append(item["name"])
        else:
            val_ids.append(item["name"].item())

    pkl.dump(train_ids, open(f"{args.savedir}/train_ids.pkl", "wb"))
    pkl.dump(val_ids, open(f"{args.savedir}/val_ids.pkl", "wb"))

    early_stop = [args.early_stop_val, args.early_stop_train]

    # Turn off gradient
    if details["multifidelity"]:
        for param in model.fn_target.conv_to_fc.parameters():
            param.requires_grad = False

    # set Trainer
    trainer = Trainer(
        model_path=args.savedir,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=val_loader,
        run_wandb=args.wandb,
        normalizer=normalizer,
    )
    # Train
    _ = trainer.train(
        device=args.device,
        start_epoch=args.start_epoch,
        n_epochs=args.epochs,
        best_loss=best_loss,
        best_metric=best_metric,
        early_stop=early_stop,
    )
    # Test results
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
    )
    test_targets = []
    test_preds = []

    best_checkpoint = torch.load(f"{args.savedir}/best_model.pth.tar")
    model.load_state_dict(best_checkpoint["state_dict"])

    (
        test_preds,
        test_targets,
        test_ids,
        metric_out,
        test_preds_fidelity,
        test_targets_fidelity,
    ) = test_model(
        model=model,
        test_loader=test_loader,
        metric_fn=metric_fn,
        device="cpu",
        # normalizer=normalizer,
        multifidelity=details["multifidelity"],
    )
    print(f"TEST Accuracy: {metric_out}")
    # Save Test Results
    pkl.dump(test_ids, open(f"{args.savedir}/test_ids.pkl", "wb"))
    pkl.dump(test_preds, open(f"{args.savedir}/test_preds.pkl", "wb"))
    pkl.dump(test_targets, open(f"{args.savedir}/test_targs.pkl", "wb"))
    if details["multifidelity"]:
        pkl.dump(
            test_preds_fidelity, open(f"{args.savedir}/test_preds_fidelity.pkl", "wb")
        )
        pkl.dump(
            test_targets_fidelity, open(f"{args.savedir}/test_targs_fidelity.pkl", "wb")
        )

    # save wandb artifacts
    if args.wandb:
        save_artifacts(args.savedir, details["multifidelity"])


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    # TODO CUDA Settings confusing
    if args.device == "cuda":
        # os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        assert torch.cuda.is_available(), "cuda is not available"

    main(args)
