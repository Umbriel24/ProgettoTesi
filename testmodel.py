import torch
import torchvision.transforms as transforms
import config

from torch.utils.data.dataloader import DataLoader
from PIL import Image
from models import MultiTaskPetModel
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data


def TestModello():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso del device: {device}")

    model = MultiTaskPetModel(backbone_name=config.BACKBONE, pretrained=False).to(device)

    model_path = "best_model.pth"

    # Carica il modello salvato e utilizza la gpu
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict=state_dict)

    model = model.to(device=device)

    model.eval()

    print("Modello caricato. Pronto per l'utilizzo")


    # Definisci le stesse trasformazioni usate in validazione (es. ridimensionamento e normalizzazione)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

        # 4. DATALOADER
    print("Configurazione dataLoader")
    num_worker = 0

    test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=num_worker,
    pin_memory=False
    )


    # VALUTAZIONE TEST SET
    correct_micro = 0
    correct_macro = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets_micro = batch["micro_label"].to(device)
            targets_macro = batch["macro_label"].to(device)

            # forward pass
            outputs_micro, outputs_macro = model(images)

            # 3. Ottieni la classe con il punteggio più alto per ciascuna testa
            _, preds_micro = torch.max(outputs_micro, dim=1)
            _, preds_macro = torch.max(outputs_macro, dim=1)

            # 4. Aggiorna i contatori confrontando le predizioni con i target reali
            correct_micro += (preds_micro == targets_micro).sum().item()
            correct_macro += (preds_macro == targets_macro).sum().item()

            # Incrementa il numero totale di campioni analizzati
            total_samples += targets_micro.size(0)

    # 5. Calcola le percentuali finali di accuratezza
    accuracy_micro = (correct_micro / total_samples) * 100
    accuracy_macro = (correct_macro / total_samples) * 100

    print("Risultati TEST SET")
    print(f"Campioni totali: {total_samples}")
    print(f"Micro-categoria (Razza): {accuracy_micro:.2f}% ({correct_micro}/{total_samples} corretti)")
    print(f"Macro-categoria (Cane/Gatto): {accuracy_macro:.2f}% ({correct_macro}/{total_samples} corretti)")


TestModello()
