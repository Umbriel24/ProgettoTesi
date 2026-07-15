import csv
import random
import torch

import config

from torch.utils.data.dataloader import DataLoader

from models import MultiTaskPetModel
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


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
    _seed=int(random.randrange(0, 10000))

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
        print("Splitting completo:")
        print(f"Campioni TRAIN: {len(train_subset)}")
        print(f"Campioni VAL: {len(val_subset)}")
        print(f"Campioni TEST: {len(test_subset)}")
    except Exception as e:
        print(f"Errore durante lo splitting: {e}")
        return

        # 3. CREAZIONE DATASET
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


    # VALUTAZIONE TEST SET
    NUM_MICRO = 37
    NUM_MACRO = 2

    correct_micro = 0
    correct_macro = 0
    total_samples = 0

    tp_micro = torch.zeros(NUM_MICRO, device=device)
    true_micro = torch.zeros(NUM_MICRO, device=device)
    tp_macro = torch.zeros(NUM_MACRO, device=device)
    true_macro = torch.zeros(NUM_MACRO, device=device)

    fp_micro = torch.zeros(NUM_MICRO, device=device)
    fp_macro = torch.zeros(NUM_MACRO, device=device)

    # Liste per sklearn (raccoglieranno tutti i campioni)
    all_preds_micro = []
    all_targets_micro = []
    all_preds_macro = []
    all_targets_macro = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets_micro = batch["micro_label"].to(device)
            targets_macro = batch["macro_label"].to(device)

            outputs_micro, outputs_macro = model(images)

            _, preds_micro = torch.max(outputs_micro, dim=1)
            _, preds_macro = torch.max(outputs_macro, dim=1)

            # Accumulo per sklearn (converti in cpu e numpy)
            all_preds_micro.extend(preds_micro.cpu().numpy())
            all_targets_micro.extend(targets_micro.cpu().numpy())
            all_preds_macro.extend(preds_macro.cpu().numpy())
            all_targets_macro.extend(targets_macro.cpu().numpy())

            correct_micro += (preds_micro == targets_micro).sum().item()
            correct_macro += (preds_macro == targets_macro).sum().item()
            total_samples += targets_micro.size(0)

            for c in range(NUM_MICRO):
                tp_micro[c] += ((preds_micro == c) & (targets_micro == c)).sum().item()
                true_micro[c] += (targets_micro == c).sum().item()
                # falsi positivi
                fp_micro[c] += ((preds_micro == c) & (targets_micro != c)).sum().item()


            for c in range(NUM_MACRO):
                tp_macro[c] += ((preds_macro == c) & (targets_macro == c)).sum().item()
                true_macro[c] += (targets_macro == c).sum().item()
                # falsi positivi:
                fp_macro[c] += ((preds_macro == c) & (targets_macro != c)).sum().item()

    print("\n--- VERIFICA sklearn ---")
    print("\nVerifica via classification_report (Razze):\n")
    #print(classification_report(all_targets_micro, all_preds_micro))

    report = classification_report(all_targets_micro, all_preds_micro, output_dict=True, zero_division= 0.0)

    # Aggiungi il seed al report
    seed = _seed
    report['seed'] = seed

    # Scrivi su CSV
    with open("report_razze.csv", "a", newline="") as csvfile:
        fieldnames = ['Classe', 'Precision', 'Recall', 'F1-score', 'Support']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Scrivi ogni classe
        for classe, metrics in report.items():
            if classe == 'seed':
                continue  # lo saltiamo qui, lo mettiamo dopo
            if isinstance(metrics, dict):
                writer.writerow({
                    'Classe': classe,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-score': f"{metrics['f1-score']:.4f}",
                    'Support': metrics['support']
                })

        # Scrivi il seed come ultima riga
        writer.writerow({
            'Classe': f"seed_{seed}",
            'Precision': '',
            'Recall': '',
            'F1-score': '',
            'Support': ''
        })

    print(f"Report salvato in report_razze.csv (seed={seed})")

for i in range(5):
    TestModello()
