import csv

import config
import torch

from models import MultiTaskPetModel
from trainer import train_one_epoch
from trainer import evaluate

from torch import nn
from torch.utils.data.dataloader import DataLoader
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    print("TEST PIPELINE INGESTION DATI \n")

    # 1. PARSING
    print("1. PARSING: \n")
    try:
        parsed_data = parse_annotation_file()
        print(f"PARSING COMPLETATO. Estratti {len(parsed_data)} campioni")
    except Exception as e:
        print(f" Errore durante il parsing: {e}")
        return

    # 2. SPLITTING
    print("\n SPLITTING IN CORSO...")
    try:
        train_subset, val_subset, test_subset = split_parsed_data(
            parsed_data=parsed_data,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
            seed=config.SEED
        )
        print("Splitting completo:")
        print(f"Campioni TRAIN: {len(train_subset)}")
        print(f"Campioni VAL: {len(val_subset)}")
        print(f"Campioni TEST: {len(test_subset)}")
    except Exception as e:
        print(f"Errore durante lo splitting: {e}")
        return


    # 3. CREAZIONE DATASET
    print("Creazione dataset per pytorch")
    train_dataset = PetDataset(data_list=train_subset, transform=config.TRAIN_TRANSFORMS)
    val_dataset = PetDataset(data_list=val_subset, transform=config.VAL_TEST_TRANSFORMS)
    test_dataset = PetDataset(data_list=test_subset, transform=config.VAL_TEST_TRANSFORMS)
    # Dataset Istanziati
    print("Dataset creati correttamente")


    # 4. DATALOADER
    print("Configurazione dataLoader")

    num_worker = 0

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
    model = MultiTaskPetModel(backbone_name=config.BACKBONE, pretrained=config.PRETRAINED).to(device)
    print("Creazione modello completa")

    # 7. Loss e Ottimizzazione
    criterion_micro = nn.CrossEntropyLoss()
    criterion_macro = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= config.LEARNING_RATE)

    print("Creazione loss e Optimizer completa")

    # 8. Training loop
    best_val_loss = float("inf")
    history = []
    patience = 6

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_micro, criterion_macro, device)
        val_loss = float(evaluate(model, val_loader, criterion_micro, criterion_macro, device))

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f" Epoca: {epoch+1} / {config.NUM_EPOCHS}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience = 0
            print(f"Modello salvato: loss è {val_loss:.4f}")
        else:
            patience += 1

        if patience > 6:
            print("6 epoche in cui non è aumentata la performance. Fine ciclo di addestramento")
            break

    print("Training completo")

    # Scrittura in CSV
    fieldnames = ["epoch", "train_loss", "val_loss"]

    with open("history.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

if __name__ == "__main__":
    main()
