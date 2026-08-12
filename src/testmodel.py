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
from DatasetLibrary.dataset_cifar import parse_cifar100_kaggle, Cifar100Dataset
from sklearn.metrics import classification_report


def TestModello(model_path, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso del device: {device}")

    # 1. PARSING DEL NOME FILE PER ESTRARRE METADATI
    _filename = os.path.basename(str(model_path))
    _stem = _filename.replace(".pth", "").replace(".pt", "")
    _parts = _stem.split("_")

    _name = _parts[1]

    # Logica per supportare sia i vecchi modelli (Pets) sia i nuovi (Pets/Cifar)
    if "percentage" in _parts[2]:
        dataset_name = "pets"
        percentage_drop = int(_parts[2].replace("percentage", ""))
        typeofdrop = _parts[3]
    else:
        dataset_name = _parts[2]
        percentage_drop = int(_parts[3].replace("percentage", ""))
        typeofdrop = _parts[4]

    # Estrazione sicura del seed dal nome del file (è sempre l'ultimo pezzo)
    try:
        _seed = int(_parts[-1])
    except ValueError:
        _seed = int(random.randrange(0, 10000))

    if seed != 0:
        _seed = seed

    print(
        f"\nValutazione Modello: {_name} | Dataset: {dataset_name} | Drop: {percentage_drop}% ({typeofdrop}) | Seed: {_seed}")

    # 2. CARICAMENTO DATI E SPLITTING (BIVIO DATASET)
    print("PARSING & SPLITTING IN CORSO...")
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
            NUM_MICRO = 37
            NUM_MACRO = 2

        elif dataset_name == "cifar100":
            cifar_path = getattr(config, 'CIFAR_PATH', "/kaggle/input/cifar-100-python/cifar-100-python")
            (train_data_full, images_train), (test_data, images_test) = parse_cifar100_kaggle(cifar_path)
            train_subset, val_subset, _ = split_parsed_data(
                parsed_data=train_data_full,
                train_ratio=0.85,
                val_ratio=0.15,
                test_ratio=0.0,
                seed=_seed
            )
            test_subset = test_data
            target_macro_class = '0'
            NUM_MICRO = 100
            NUM_MACRO = 20
            parsed_data = []  # Placeholder per Cifar
        else:
            print(f"Dataset {dataset_name} non riconosciuto.")
            return

    except Exception as e:
        print(f"Errore durante lo splitting: {e}")
        return

    # Usiamo il dropper SOLO per scoprire quali classi NON sono state viste in training
    dropped_breeds = set()
    if typeofdrop == "micro" and percentage_drop > 0:
        dropper = DatasetDropper(train_subset, seed=_seed)
        dropper.drop_micro(target_macro=target_macro_class, percentage=percentage_drop / 100)
        dropped_breeds = dropper.dropped_micro_ids
        print(f"Classi ignote al modello ({len(dropped_breeds)}): {sorted(dropped_breeds)}")

    print(f"Campioni TEST (Mondo Reale intatto): {len(test_subset)}")

    # 3. INIZIALIZZAZIONE MODELLO
    model = ModelsCreator(backbone_name=_name, num_micro_classes=NUM_MICRO, num_macro_classes=NUM_MACRO,
                          pretrained=False).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    print("Modello caricato. Pronto per l'utilizzo")

    # 4. CREAZIONE DATASET E DATALOADER
    print("Configurazione dataLoader")
    if dataset_name == "pets":
        test_dataset = PetDataset(data_list=test_subset, transform=config.VAL_TEST_TRANSFORMS)
    elif dataset_name == "cifar100":
        test_dataset = Cifar100Dataset(data_list=test_subset, image_arrays=images_test,
                                       transform=config.VAL_TEST_TRANSFORMS)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    all_preds_micro = []
    all_targets_micro = []

    # 5. INFERENZA
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets_micro = batch["micro_label"].to(device)

            outputs_micro, outputs_macro = model(images)
            _, preds_micro = torch.max(outputs_micro, dim=1)

            all_preds_micro.extend(preds_micro.cpu().numpy())
            all_targets_micro.extend(targets_micro.cpu().numpy())

    print("\n--- VERIFICA sklearn ---")

    active_labels = list(range(NUM_MICRO))
    report = classification_report(all_targets_micro, all_preds_micro, labels=active_labels, output_dict=True,
                                   zero_division=0.0)

    # Calcolo della Metrica di Specializzazione (Solo Classi Note)
    known_f1_scores = []
    for c in range(NUM_MICRO):
        # Pets aveva indici originali che partivano da 1, CIFAR parte da 0
        original_class_id = str(c + 1) if dataset_name == "pets" else str(c)
        if original_class_id not in dropped_breeds:
            known_f1_scores.append(report[str(c)]['f1-score'])

    known_macro_f1 = sum(known_f1_scores) / len(known_f1_scores) if known_f1_scores else 0.0
    print(f"Macro F1-Score GLOBALE (Open World): {report['macro avg']['f1-score']:.4f}")
    print(f"Macro F1-Score CLASSI NOTE (Specializzazione): {known_macro_f1:.4f}")

    # Creazione mappatura nomi per il CSV
    idx_to_class = {}
    if dataset_name == "pets":
        for img_path, micro_lbl, _ in parsed_data:
            filename = os.path.basename(str(img_path))
            breed_name = filename.rsplit('_', 1)[0]
            idx_str = str(int(micro_lbl) - 1)
            idx_to_class[idx_str] = breed_name
    elif dataset_name == "cifar100":
        for i in range(100):
            idx_to_class[str(i)] = f"Cifar_Class_{i}"

    # --- 6. SCRITTURA REPORT CLASSI ---
    csv_classi = config.PERSISTANCE_PATH / f"report_{_name}_{dataset_name}_classi.csv"
    file_exists_classi = csv_classi.exists()

    with open(csv_classi, "a", newline="") as csvfile:
        fieldnames = ['drop_type', 'drop_percentage', 'seed', 'Classe', 'Vista_In_Train', 'Precision', 'Recall',
                      'F1-score', 'Support']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists_classi:
            writer.writeheader()

        for classe_idx, metrics in report.items():
            if classe_idx in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(metrics, dict):
                nome_classe = idx_to_class.get(classe_idx, f"Classe_{classe_idx}")
                original_class_id = str(int(classe_idx) + 1) if dataset_name == "pets" else str(classe_idx)
                vista = "No" if original_class_id in dropped_breeds else "Si"

                writer.writerow({
                    'drop_type': typeofdrop,
                    'drop_percentage': percentage_drop,
                    'seed': _seed,
                    'Classe': nome_classe,
                    'Vista_In_Train': vista,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-score': f"{metrics['f1-score']:.4f}",
                    'Support': metrics['support']
                })

    # --- 7. SCRITTURA REPORT GLOBALE ---
    csv_globali = config.PERSISTANCE_PATH / f"report_{_name}_{dataset_name}_globali.csv"
    file_exists_glob = csv_globali.exists()

    with open(csv_globali, "a", newline="") as csvfile:
        fieldnames_glob = ['drop_type', 'drop_percentage', 'seed', 'Accuracy', 'Macro_F1_Globale', 'Macro_F1_Note']
        writer_glob = csv.DictWriter(csvfile, fieldnames=fieldnames_glob)

        if not file_exists_glob:
            writer_glob.writeheader()

        writer_glob.writerow({
            'drop_type': typeofdrop,
            'drop_percentage': percentage_drop,
            'seed': _seed,
            'Accuracy': f"{report.get('accuracy', 0.0):.4f}",
            'Macro_F1_Globale': f"{report['macro avg']['f1-score']:.4f}",
            'Macro_F1_Note': f"{known_macro_f1:.4f}"
        })

    print(
        f"Report salvato per {_name} (dataset: {dataset_name}, drop: {percentage_drop}%, tipo: {typeofdrop}, seed: {_seed})\n")