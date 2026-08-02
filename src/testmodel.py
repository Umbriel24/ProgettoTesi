import csv
import os
import random
import torch

import config

from torch.utils.data.dataloader import DataLoader

from modelscreator import ModelsCreator
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data
from DatasetLibrary.dataset_dropper import DatasetDropper
from sklearn.metrics import classification_report


def TestModello(model_path, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso del device: {device}")

    _filename = os.path.basename(str(model_path))
    _stem = _filename.replace(".pth", "").replace(".pt", "")
    _parts = _stem.split("_")

    _name = _parts[1]
    percentage_drop = 0
    typeofdrop = "macro"
    try:
        percentage_drop = int(_parts[2].replace("percentage", ""))
        typeofdrop = _parts[3]
    except (IndexError, ValueError):
        print("Nome modello senza info di drop: valuto sul test set completo.")

    _seed = int(random.randrange(0, 10000))
    if seed != 0:
        _seed = seed

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
            seed=_seed
        )
    except Exception as e:
        print(f"Errore durante lo splitting: {e}")
        return

    # Usiamo il dropper SOLO per scoprire quali razze NON sono state viste in training
    dropped_breeds = set()
    if typeofdrop == "micro" and percentage_drop > 0:
        target_macro_class = '2'
        dropper = DatasetDropper(train_subset, seed=_seed)
        dropper.drop_micro(target_macro=target_macro_class, percentage=percentage_drop / 100)
        dropped_breeds = dropper.dropped_micro_ids
        print(f"Razze ignote al modello ({len(dropped_breeds)}): {sorted(dropped_breeds)}")

    # ATTENZIONE: IL TEST SET NON VIENE TOCCATO! Resta al 100% per valutare l'Open World
    print(f"Campioni TEST (Mondo Reale intatto): {len(test_subset)}")

    NUM_MICRO = 37
    NUM_MACRO = 2

    # Inizializzazione Modello con 37 classi fisse
    model = ModelsCreator(backbone_name=_name, num_micro_classes=NUM_MICRO, num_macro_classes=NUM_MACRO,
                          pretrained=False).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    print("Modello caricato. Pronto per l'utilizzo")

    # 3. CREAZIONE DATASET (Senza mapping dinamico)
    print("Creazione dataset per pytorch")
    _ = PetDataset(data_list=train_subset, transform=config.TRAIN_TRANSFORMS)
    _ = PetDataset(data_list=val_subset, transform=config.VAL_TEST_TRANSFORMS)
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

    all_preds_micro = []
    all_targets_micro = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets_micro = batch["micro_label"].to(device)

            outputs_micro, outputs_macro = model(images)
            _, preds_micro = torch.max(outputs_micro, dim=1)

            all_preds_micro.extend(preds_micro.cpu().numpy())
            all_targets_micro.extend(targets_micro.cpu().numpy())

    print("\n--- VERIFICA sklearn ---")

    # Forziamo le labels da 0 a 36 per assicurarci che Sklearn non ometta le razze mai predette
    active_labels = list(range(NUM_MICRO))
    report = classification_report(all_targets_micro, all_preds_micro, labels=active_labels, output_dict=True,
                                   zero_division=0.0)

    # Calcolo della Metrica di Specializzazione: F1-Score solo sulle classi NOTE
    known_f1_scores = []
    for c in range(NUM_MICRO):
        original_class_id = str(c + 1)  # Nel dataset originale le label partivano da 1
        if original_class_id not in dropped_breeds:
            known_f1_scores.append(report[str(c)]['f1-score'])

    known_macro_f1 = sum(known_f1_scores) / len(known_f1_scores) if known_f1_scores else 0.0
    print(f"Macro F1-Score GLOBALE (Open World): {report['macro avg']['f1-score']:.4f}")
    print(f"Macro F1-Score CLASSI NOTE (Specializzazione): {known_macro_f1:.4f}")

    # Scrivi su CSV: Unico file per architettura (es. report_resnet18.csv)
    csv_path = config.PERSISTANCE_PATH / f"report_{_name}.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as csvfile:
        # Aggiungiamo le colonne di metadati per riconoscere i test nel file unico
        fieldnames = ['drop_type', 'drop_percentage', 'seed', 'Classe', 'Precision', 'Recall', 'F1-score', 'Support']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # Scrivi ogni classe
        for classe, metrics in report.items():
            if classe in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(metrics, dict):
                writer.writerow({
                    'drop_type': typeofdrop,
                    'drop_percentage': percentage_drop,
                    'seed': _seed,
                    'Classe': classe,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-score': f"{metrics['f1-score']:.4f}",
                    'Support': metrics['support']
                })

        # Scrivi la doppia metrica come ultime righe del blocco
        writer.writerow({
            'drop_type': typeofdrop,
            'drop_percentage': percentage_drop,
            'seed': _seed,
            'Classe': "macro_avg_globale",
            'Precision': '', 'Recall': '',
            'F1-score': f"{report['macro avg']['f1-score']:.4f}",
            'Support': ''
        })
        writer.writerow({
            'drop_type': typeofdrop,
            'drop_percentage': percentage_drop,
            'seed': _seed,
            'Classe': "macro_avg_note",
            'Precision': '', 'Recall': '',
            'F1-score': f"{known_macro_f1:.4f}",
            'Support': ''
        })
        # Rimosso il row "seed_{_seed}" singolo, perché ora il seed è integrato in ogni riga

    print(f"Report salvato (seed={_seed})")
