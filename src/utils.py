import pickle
from pathlib import Path

import torch


def save_checkpoint(output_dir, step, epoch, dataloader, model, optimizer, scheduler):
    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Save the checkpoint
    state_dict = {
        "step_epoch": (step, epoch),
        "dataloader_state_dict": dataloader.get_state_dict(),
        "model_state_dict": model.get_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()

    with open(output_dir / "checkpoint.chkpt", "wb") as outfile:
        torch.save(state_dict, outfile, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(output_dir: Path, device="cpu"):
    # Load the latest checkpoint file
    infile = output_dir / "checkpoint.chkpt"
    if not infile.exists():
        raise FileNotFoundError("Checkpoints not found.")

    checkpoint = torch.load(infile, map_location=device)

    # Load the checkpoint values
    step, epoch = checkpoint["step_epoch"]
    dataloader_state_dict = checkpoint["dataloader_state_dict"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)

    # Return the loaded values
    return (
        step,
        epoch,
        dataloader_state_dict,
        model_state_dict,
        optimizer_state_dict,
        scheduler_state_dict,
    )


def get_num_param_and_model_size(model):
    print("*" * 35)
    num_params = sum(p.nelement() for p in model.parameters())
    num_trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print("Total Params           : {}".format(num_params))
    print("Total Trainable Params : {}".format(num_trainable))

    num_buffers = sum(b.nelement() for b in model.buffers())
    print("Total Buffers          : {}".format(num_buffers))

    size_params = sum(p.nelement() * p.element_size() for p in model.parameters())
    size_buffers = sum(p.nelement() * p.element_size() for p in model.buffers())
    size_all_mb = (size_params + size_buffers) / 1024**2
    print("Model size             : {:.3f}MB".format(size_all_mb))
    print("*" * 35)


def classification_metrics(y_true, y_pred):  
    assert y_true.shape == y_pred.shape, "y_true, y_pred shapes don't match" 
    # Calculate accuracy
    accuracy = torch.sum(y_true == y_pred)/len(y_true)
    
    # Calculate precision, recall, and F1-score for each class
    unique_classes = torch.unique(torch.concatenate((y_true, y_pred)))
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for class_label in unique_classes:
        true_positive = torch.sum((y_true == class_label) & (y_pred == class_label))
        false_positive = torch.sum((y_true != class_label) & (y_pred == class_label))
        false_negative = torch.sum((y_true == class_label) & (y_pred != class_label))
        
        precision = true_positive / (true_positive + false_positive + 1e-7)
        recall = true_positive / (true_positive + false_negative + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Calculate the macro-average precision, recall, and F1-score
    precision_macro = torch.mean(torch.Tensor(precision_scores))
    recall_macro = torch.mean(torch.Tensor(recall_scores))
    f1_macro = torch.mean(torch.Tensor(f1_scores))
    
    return accuracy.item(), precision_macro.item(), recall_macro.item(), f1_macro.item()