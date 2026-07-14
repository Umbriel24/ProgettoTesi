import torch
import config

def train_one_epoch(model, loader, optimizer, criterion_micro, criterion_macro, device):
    model.train() # Addestramento
    total_loss = 0.0

    for batch in loader: # Per ogni batch di immagini
        images = batch["image"].to(device)
        micro_labels = batch["micro_label"].to(device)
        macro_labels = batch["macro_label"].to(device)

        # Azzeriamo i gradienti
        optimizer.zero_grad()

        # forward pass
        out_micro, out_macro = model(images)

        # calcolo loss, ovvero perdita
        loss_micro = criterion_micro(out_micro, micro_labels)
        loss_macro = criterion_macro(out_macro, macro_labels)

        loss = config.ALPHA * loss_micro + config.BETA * loss_macro


        # backward pass e upgrade pesi
        loss.backward() # backpropagation
        optimizer.step() # aggiornamento pesi

        total_loss += loss.item()

    return total_loss / len(loader)



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
