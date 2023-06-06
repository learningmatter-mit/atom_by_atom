import wandb


def upload_artifact(name, filename, artifact_type):
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_file(filename)
    wandb.run.log_artifact(artifact)


def save_artifacts(savedir, multifidelity):
    model_filename = savedir + "/best_model.pth.tar"
    test_ids_filename = savedir + "/test_ids.pkl"
    test_preds_filename = savedir + "/test_preds.pkl"
    test_targs_filename = savedir + "/test_targs.pkl"
    if multifidelity:
        test_preds_fidelity_filename = savedir + "/test_preds_fidelity.pkl"
        test_targs_fidelity_filename = savedir + "/test_targs_fidelity.pkl"
    upload_artifact("model", model_filename, artifact_type="model")
    upload_artifact("test_ids", test_ids_filename, artifact_type="test_ids")
    upload_artifact("test_preds", test_preds_filename, artifact_type="test_preds")
    upload_artifact("test_targs", test_targs_filename, artifact_type="test_targs")
    if multifidelity:
        upload_artifact(
            "test_preds_fidelity",
            test_preds_fidelity_filename,
            artifact_type="test_preds_fidelity",
        )
        upload_artifact(
            "test_targs_fidelity",
            test_targs_fidelity_filename,
            artifact_type="test_targs_fidelity",
        )
