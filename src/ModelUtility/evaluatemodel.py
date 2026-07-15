import torch
import config

def evaluate(model, loader, criterion_micro, criterion_macro, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            micro_labels = batch["micro_label"].to(device)
            macro_labels = batch["macro_label"].to(device)

            out_micro, out_macro = model(images)

            loss_micro = criterion_micro(out_micro, micro_labels)
            loss_macro = criterion_macro(out_macro, macro_labels)

            loss = config.ALPHA * loss_micro + config.BETA * loss_macro

            total_loss += loss.item()

    return total_loss / len(loader)
