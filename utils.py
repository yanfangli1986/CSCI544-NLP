import torch


def save_model_object(model):
    torch.save({'state_dict': model.state_dict()}, "best_model.pt")
    return
