import torch
import os
from src.utils.trainer_utils import EarlyStopping

def test_early_stopping_logic(tmp_path):
    model = torch.nn.Linear(10, 2)
    save_path = os.path.join(tmp_path, "best_model.pt")
    
    es = EarlyStopping(patience=2, min_delta=0.1, save_path=save_path)
    
    # Epoch 1: Loss = 1.0 (Best)
    es(1.0, model)
    assert not es.early_stop
    assert es.counter == 0
    assert os.path.exists(save_path)
    
    # Epoch 2: Loss = 0.95 (Didn't improve by 0.1 delta)
    es(0.95, model)
    assert not es.early_stop
    assert es.counter == 1
    
    # Epoch 3: Loss = 0.95 (Didn't improve again)
    es(0.95, model)
    assert es.early_stop, "Early stopping should trigger after patience=2 is exhausted"
    assert es.counter == 2
