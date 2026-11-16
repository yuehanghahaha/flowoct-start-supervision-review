"""Final-report inference using the released generator; no replacement training losses."""
import sys
import torch
from experiment import ROOT
sys.path.insert(0, str(ROOT / "source"))
from models.networks_2g_st import generator3D


def inference_generator(state, device):
    model = generator3D(64)
    model.load_state_dict(state)
    return model.to(device=device, dtype=torch.float32).eval()
