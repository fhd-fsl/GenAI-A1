import wandb
import os
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any] = None):
        """
        Initializes the Weights & Biases experiment tracker.
        
        Args:
            project_name: Name of the W&B project.
            run_name: Name of this specific run (e.g., 'task1_trial_4').
            config: Dictionary of hyperparameters to log.
        """
        self.project_name = project_name
        self.run_name = run_name
        # Initialize wandb run
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config
        )

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Logs a dictionary of metrics to W&B."""
        wandb.log(metrics, step=step)

    def log_model(self, model_path: str, model_name: str):
        """Logs a saved model artifact to W&B."""
        artifact = wandb.Artifact(name=model_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def log_images(self, tag: str, images: list, step: int = None):
        """
        Logs a list of images to W&B.
        Args:
            images: List of PIL Images, numpy arrays, or torch tensors.
        """
        wandb_images = [wandb.Image(img) for img in images]
        wandb.log({tag: wandb_images}, step=step)

    def finish(self):
        """Ends the W&B run."""
        wandb.finish()
