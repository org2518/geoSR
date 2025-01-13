import wandb


def download_checkpoint(
    checkpoint_reference: str,
    project: str = "GeoSR",
    root: str = "artifacts",
) -> str:
    run = wandb.init(project="GeoSR")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download(root=root)
    return artifact_dir
