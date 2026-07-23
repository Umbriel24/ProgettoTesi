import glob
import os

import torch
from sympy import false

import config
import csv

from modelscreator import ModelsCreator
from ModelUtility.evaluate_model import evaluate_model

from torch import nn
from torch.utils.data.dataloader import DataLoader
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data
from DatasetLibrary.dataset_dropper import DatasetDropper


def create_and_train_model(type_of_net: str, pre_trained_value: bool, percentage_drop: int, typeofdrop: str = "macro"):
    # Set dei seed.
    _seed = config.SEED
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    t_modelname = f"model_{type_of_net}_percentage{percentage_drop}_{typeofdrop}_{_seed}.pt"
    if check_model_existence(t_modelname):
        print("Il modello esiste. skip training ")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    print("TEST PIPELINE INGESTION DATI \n")

    # 1. PARSING
    print("1. PARSING: \n")
    try:
        parsed_data = parse_annotation_file()
        print(f"PARSING COMPLETATO. Estratti {len(parsed_data)} campioni")
    except Exception as e:
        print(f" Errore durante il parsing: {e}")
        return

    # 2. SPLITTING dati nei 3 gruppi
    print("\n SPLITTING IN CORSO...")
    try:
        train_subset, val_subset, test_subset = split_parsed_data(
            parsed_data=parsed_data,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
            seed=_seed
        )
        print("Splitting completo:")
        print(f"Campioni TRAIN: {len(train_subset)}")
        print(f"Campioni VAL: {len(val_subset)}")
        print(f"Campioni TEST: {len(test_subset)}")
    except Exception as e:
        print(f"Errore durante lo splitting: {e}")
        return

    target_macro_class = '2'
    drop_percentage = percentage_drop

    dropper = DatasetDropper(train_subset, seed=_seed)

    if typeofdrop == "micro":
        train_subset = dropper.drop_micro(target_macro=target_macro_class, percentage=drop_percentage / 100)
    else:
        train_subset = dropper.drop_macro(target_macro=target_macro_class, percentage=drop_percentage / 100)

    print(f"Campioni TRAINING DOPO IL DROP ({drop_percentage}%): {len(train_subset)}")

    # 3. CREAZIONE DATASET dei 3 gruppi
    print("Creazione dataset per pytorch")
    train_dataset = PetDataset(data_list=train_subset, transform=config.TRAIN_TRANSFORMS)
    val_dataset = PetDataset(data_list=val_subset, transform=config.VAL_TEST_TRANSFORMS)
    _ = PetDataset(data_list=test_subset, transform=config.VAL_TEST_TRANSFORMS)
    # Dataset Istanziati
    print("Dataset creati correttamente")

    # 4. DATALOADER
    print("Configurazione dataLoader")

    num_worker = os.cpu_count()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=False
    )

    # 5. Verifica primo batch del train_loader
    print("Test first batch train_loader")

    first_batch = next(iter(train_loader))

    images = first_batch["image"]
    micro_label = first_batch["micro_label"]
    macro_label = first_batch["macro_label"]

    print(f"Forma tensore immagini: {images.shape}")
    print(f"Forma tensore Micro-label: {micro_label.shape}")
    print(f"Forma tensore Macro-label: {macro_label.shape}")
    print(f"Tipo dato immagini: {images.dtype}")
    print(f"Tipo micro_label: {micro_label.dtype}")
    print(f"Tipo macro_label: {macro_label.dtype}")

    # 6. Modello
    model = ModelsCreator(backbone_name=type_of_net, pretrained=pre_trained_value).to(device)
    print("Creazione modello completa")

    # 7. Loss e Ottimizzazione
    criterion_micro = nn.CrossEntropyLoss()
    criterion_macro = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Creazione loss e Optimizer completa")

    # 8. Training loop
    best_val_loss = float("inf")
    history = []
    patience = 6

    # Addestra su 50 epoche. Se in 6 epoche di seguito non migliora, si ferma.
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_micro, criterion_macro, device)
        val_loss = float(evaluate_model(model, val_loader, criterion_micro, criterion_macro, device))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        print(f" Epoca: {epoch + 1} / {config.NUM_EPOCHS}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            print(f"Miglior modello ora ha val_loss: {val_loss:.4f}")
        else:
            patience += 1

        if patience > 6:
            print("6 epoche in cui non è aumentata la performance. Fine ciclo di addestramento")
            break

    print("Training completo")

    # Scrittura in CSV
    save_model(model, type_of_net, percentage_drop, _seed, typeofdrop)
    save_history(history, type_of_net, percentage_drop, _seed, typeofdrop)


# metodo che traina per un'epoca.
def train_one_epoch(model, loader, optimizer, criterion_micro, criterion_macro, device):
    model.train()  # Addestramento
    total_loss = 0.0

    for batch in loader:  # Per ogni batch di immagini
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
        loss.backward()  # backpropagation
        optimizer.step()  # aggiornamento pesi

        total_loss += loss.item()

    return total_loss / len(loader)


def save_model(model, net_name, percentage_drop, seed, typeOfDrop):
    torch.save(model.state_dict(), config.PERSISTANCE_PATH / f"model_{net_name}_percentage{percentage_drop}_{typeOfDrop}_{seed}.pt")


def save_history(history, type_of_net, percentage_drop, seed, typeOfDrop):
    # Scrittura in CSV
    fieldnames = ["epoch", "train_loss", "val_loss"]

    with open(config.PERSISTANCE_PATH /  f"{type_of_net}_percentage{percentage_drop}_{typeOfDrop}_{seed}.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


# Check se il modello esiste. Se esiste returna true, altrimenti false
def check_model_existence(model_name: str):
        for dirName, subdirList, fileList in os.walk(config.PERSISTANCE_PATH):
            for fname in fileList:
                if fname == model_name:
                    return True
        return False