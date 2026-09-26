import torch
import os

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the model checkpoint automatically when validation loss decreases.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, save_path: str = "checkpoints/best_model.pt"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
