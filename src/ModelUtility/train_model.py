import os
import random
from collections import defaultdict

import torch
import config
import csv

from modelscreator import ModelsCreator
from ModelUtility.evaluate_model import evaluate_model
from torch import nn
from torch.utils.data.dataloader import DataLoader

# Import per Pets
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file

# Import per Cifar-100
from DatasetLibrary.dataset_cifar import parse_cifar100_kaggle, Cifar100Dataset

from DatasetLibrary.dataset_splitter import split_parsed_data
from DatasetLibrary.dataset_dropper import DatasetDropper


def create_and_train_model(type_of_net: str, pre_trained_value: bool, percentage_drop: int, typeofdrop: str = "macro",
                           dataset_name: str = "pets", _seed: int = config.SEED):
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    t_modelname = f"model_{type_of_net}_{dataset_name}_percentage{percentage_drop}_{typeofdrop}_{_seed}.pt"
    if check_model_existence(t_modelname):
        print(f"Il modello {t_modelname} esiste. skip training ")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        if dataset_name == "pets":
            parsed_data = parse_annotation_file()
            train_subset, val_subset, test_subset = split_parsed_data(
                parsed_data=parsed_data,
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO,
                test_ratio=config.TEST_RATIO,
                seed=_seed
            )
            target_macro_class = '2'
            num_micro_classes = 37
            num_macro_classes = 2

        elif dataset_name == "cifar100":
            # Path base in Kaggle. Assicurati che corrisponda alla tua directory!
            cifar_path = "/kaggle/input/cifar-100-python/cifar-100-python"
            if hasattr(config, 'CIFAR_PATH'):
                cifar_path = config.CIFAR_PATH

            (train_data_full, images_train), (test_data, images_test) = parse_cifar100_kaggle(cifar_path)

            # Cifar ha già il test set, splittiamo il train_full per ricavare il validation
            train_subset, val_subset, _ = split_parsed_data(
                parsed_data=train_data_full,
                train_ratio=0.85,
                val_ratio=0.15,
                test_ratio=0.0,
                seed=_seed
            )

            # Prendiamo solo 5000 immagini divise per le classi
            test_subset = test_data
            random.seed(_seed)
            class_backet = defaultdict(list)

            #1- Raggruppa per classe
            for item in train_subset:
                micro_class = item[1]
                class_backet[micro_class].append(item)

            train_subset_stratified = []
            sample_per_class = 50

            #2 peschiamo 50 campioni per classe
            for c_label, items in class_backet.items():
                train_subset_stratified.extend(random.sample(items, sample_per_class))

            #3. Shuffle
            random.shuffle(train_subset_stratified)
            train_subset = train_subset_stratified

            target_macro_class = '0'  # Drop di esempio: la macroclasse 0 (Aquatic mammals)
            num_micro_classes = 100
            num_macro_classes = 20

        else:
            print(f"Dataset {dataset_name} non supportato!")
            return

    except Exception as e:
        print(f"Errore durante il parsing/splitting: {e}")
        return

    drop_percentage = percentage_drop
    dropper = DatasetDropper(train_subset, seed=_seed)

    if typeofdrop == "micro":
        train_subset = dropper.drop_micro(target_macro=target_macro_class, percentage=drop_percentage / 100)
        dropped_breeds = dropper.dropped_micro_ids
        print(f"Classi rimosse SOLO dal training ({len(dropped_breeds)}): {sorted(dropped_breeds)}")
    else:
        train_subset = dropper.drop_macro(target_macro=target_macro_class, percentage=drop_percentage / 100)

    print(f"Campioni TRAINING DOPO IL DROP ({drop_percentage}%): {len(train_subset)}")
    print(f"Campioni VAL (Intero, mondo reale): {len(val_subset)}")

    if dataset_name == "pets":
        train_dataset = PetDataset(data_list=train_subset, transform=config.TRAIN_TRANSFORMS)
        val_dataset = PetDataset(data_list=val_subset, transform=config.VAL_TEST_TRANSFORMS)
        _ = PetDataset(data_list=test_subset, transform=config.VAL_TEST_TRANSFORMS)
    elif dataset_name == "cifar100":
        train_dataset = Cifar100Dataset(data_list=train_subset, image_arrays=images_train,
                                        transform=config.TRAIN_TRANSFORMS)
        val_dataset = Cifar100Dataset(data_list=val_subset, image_arrays=images_train,
                                      transform=config.VAL_TEST_TRANSFORMS)
        _ = Cifar100Dataset(data_list=test_subset, image_arrays=images_test, transform=config.VAL_TEST_TRANSFORMS)

    num_worker = os.cpu_count() or 0
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_worker,
                              pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_worker,
                            pin_memory=False)

    model = ModelsCreator(backbone_name=type_of_net, pretrained=pre_trained_value, num_micro_classes=num_micro_classes,
                          num_macro_classes=num_macro_classes).to(device)
    print(f"Creazione modello {type_of_net} completa")

    criterion_micro = nn.CrossEntropyLoss()
    criterion_macro = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Inizio training modello {type_of_net} su {dataset_name} ({percentage_drop}% {typeofdrop})")

    best_val_loss = float("inf")
    history = []
    patience = 6

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_micro, criterion_macro, device)
        val_loss = float(evaluate_model(model, val_loader, criterion_micro, criterion_macro, device))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoca: {epoch + 1} / {config.NUM_EPOCHS}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            print(f"Miglior modello ora ha val_loss: {val_loss:.4f}")
        else:
            patience += 1

        if patience > 6:
            print("6 epoche in cui non è aumentata la performance. Fine ciclo di addestramento")
            break

    save_model(model, type_of_net, dataset_name, percentage_drop, _seed, typeofdrop)
    save_history(history, type_of_net, dataset_name, percentage_drop, _seed, typeofdrop)


def train_one_epoch(model, loader, optimizer, criterion_micro, criterion_macro, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        micro_labels = batch["micro_label"].to(device)
        macro_labels = batch["macro_label"].to(device)

        optimizer.zero_grad()
        out_micro, out_macro = model(images)
        loss_micro = criterion_micro(out_micro, micro_labels)
        loss_macro = criterion_macro(out_macro, macro_labels)
        loss = config.ALPHA * loss_micro + config.BETA * loss_macro

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def save_model(model, net_name, dataset_name, percentage_drop, seed, typeOfDrop):
    torch.save(model.state_dict(),
               config.PERSISTANCE_PATH / f"model_{net_name}_{dataset_name}_percentage{percentage_drop}_{typeOfDrop}_{seed}.pt")


def save_history(history, type_of_net, dataset_name, percentage_drop, seed, typeOfDrop):
    csv_path = config.PERSISTANCE_PATH / f"{type_of_net}_{dataset_name}.csv"
    file_exists = csv_path.exists()

    fieldnames = ["drop_type", "drop_percentage", "seed", "epoch", "train_loss", "val_loss"]

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for row in history:
            writer.writerow({
                "drop_type": typeOfDrop,
                "drop_percentage": percentage_drop,
                "seed": seed,
                "epoch": row["epoch"],
                "train_loss": f"{row['train_loss']:.4f}",
                "val_loss": f"{row['val_loss']:.4f}"
            })


def check_model_existence(model_name: str):
    for dirName, subdirList, fileList in os.walk(config.PERSISTANCE_PATH):
        for fname in fileList:
            if fname == model_name:
                return True
    return False