import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None

import config
from modelscreator import ModelsCreator
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data
from DatasetLibrary.dataset_dropper import DatasetDropper
from DatasetLibrary.dataset_cifar import parse_cifar100_kaggle, Cifar100Dataset


def sample_few_shot_support(train_subset_kept, dropped_samples, dropped_breeds, shots=5, seed=config.SEED):
    """
    Costruisce il dataset di supporto per il calcolo dei prototipi:
    - Per le classi note (sopravvissute al micro-drop): utilizza tutti i campioni presenti nel training set.
    - Per le classi non viste (dropped_breeds): seleziona casualmente esattamente `shots` immagini
      dal pool di campioni scartati dal training set, usando un seed deterministico.
    """
    support_rng = random.Random(seed)

    # Raggruppa i campioni scartati per micro-classe
    dropped_by_class = defaultdict(list)
    for sample in dropped_samples:
        dropped_by_class[sample[1]].append(sample)

    unseen_support = []
    for breed_id in sorted(dropped_breeds):
        pool = list(dropped_by_class[breed_id])
        support_rng.shuffle(pool)
        k_shots = min(shots, len(pool))
        unseen_support.extend(pool[:k_shots])

    # Unione: campioni noti dal train + few-shot support per le classi droppate
    full_support = list(train_subset_kept) + unseen_support
    return full_support, unseen_support


def compute_prototypes(model, support_loader, num_classes, device):
    """
    Calcola il prototipo (baricentro degli embedding) per ciascuna classe:
    c_k = (1 / |S_k|) * sum_{x in S_k} f_theta(x)
    """
    model.eval()
    feature_sums = defaultdict(lambda: None)
    feature_counts = defaultdict(int)

    with torch.no_grad():
        for batch in support_loader:
            images = batch["image"].to(device)
            micro_labels = batch["micro_label"].to(device)

            # Estrazione feature dal backbone (prima delle teste lineari)
            features = model.backbone(images)
            features = torch.flatten(features, 1)

            for feat, lbl in zip(features, micro_labels):
                c = lbl.item()
                if feature_sums[c] is None:
                    feature_sums[c] = feat.clone()
                else:
                    feature_sums[c] += feat
                feature_counts[c] += 1

    # Determinazione della dimensione dell'embedding
    first_feat = next((v for v in feature_sums.values() if v is not None), None)
    if first_feat is None:
        raise ValueError("Nessuna feature estratta dal support set.")
    feat_dim = first_feat.shape[0]

    prototypes = torch.zeros((num_classes, feat_dim), device=device)
    for c in range(num_classes):
        if feature_counts[c] > 0:
            prototypes[c] = feature_sums[c] / feature_counts[c]
        else:
            # Fallback nel caso limite di classe priva di campioni
            prototypes[c] = torch.zeros(feat_dim, device=device)

    return prototypes


def evaluate_prototypical_model(model_path, shots=5, seed=0):
    """
    Esegue la valutazione Prototypical Few-Shot Adaptation (Opzione A + Scenario A):
    1. Carica il backbone addestrato con micro-drop.
    2. Costruisce il support set (train set per classi note + `shots` campioni per classi droppate).
    3. Calcola i prototipi nello spazio delle feature per tutte le classi (Open World).
    4. Classifica il test set puro mediante distanza Euclidea dai prototipi.
    5. Calcola e salva le metriche (Globale, Classi Note, Classi Non Viste).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ProtoNet] Uso del device: {device}")

    # 1. PARSING DEL NOME FILE PER ESTRARRE I METADATI
    filename = os.path.basename(str(model_path))
    stem = filename.replace(".pth", "").replace(".pt", "")
    parts = stem.split("_")

    name = parts[1]

    if "percentage" in parts[2]:
        dataset_name = "pets"
        percentage_drop = int(parts[2].replace("percentage", ""))
        typeofdrop = parts[3]
    else:
        dataset_name = parts[2]
        percentage_drop = int(parts[3].replace("percentage", ""))
        typeofdrop = parts[4]

    try:
        model_seed = int(parts[-1])
    except ValueError:
        model_seed = int(random.randrange(0, 10000))

    active_seed = seed if seed != 0 else model_seed

    # Verifica vincolo: opera solo su modelli micro-drop con classi non viste
    if typeofdrop != "micro" or percentage_drop <= 0:
        print(f"[SKIP ProtoNet] Il file {filename} non è un modello micro-drop con classi non viste (tipo: {typeofdrop}, drop: {percentage_drop}%).")
        return False

    print(f"\n=== Valutazione Prototypical Few-Shot: {name} ===")
    print(f"Dataset: {dataset_name} | Drop: {percentage_drop}% ({typeofdrop}) | Seed: {active_seed} | Support Shots: {shots}")

    # 2. CARICAMENTO E SPLIT DEI DATI
    try:
        if dataset_name == "pets":
            parsed_data = parse_annotation_file()
            train_subset, val_subset, test_subset = split_parsed_data(
                parsed_data=parsed_data,
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO,
                test_ratio=config.TEST_RATIO,
                seed=active_seed
            )
            target_macro_class = '2'
            num_micro = 37
            num_macro = 2

        elif dataset_name == "cifar100":
            cifar_path = getattr(config, 'CIFAR_PATH', "/kaggle/input/cifar-100-python/cifar-100-python")
            (train_data_full, images_train), (test_data, images_test) = parse_cifar100_kaggle(cifar_path)
            train_subset, val_subset, _ = split_parsed_data(
                parsed_data=train_data_full,
                train_ratio=0.85,
                val_ratio=0.15,
                test_ratio=0.0,
                seed=active_seed
            )
            test_subset = test_data

            # Sincronizzazione campionamento stratificato con train
            random.seed(active_seed)
            class_buckets = defaultdict(list)
            for item in train_subset:
                class_buckets[item[1]].append(item)

            train_subset_stratified = []
            samples_per_class = config.CIFAR_SAMPLES_PER_CLASS
            for c_label, items in class_buckets.items():
                train_subset_stratified.extend(random.sample(items, samples_per_class))

            random.shuffle(train_subset_stratified)
            train_subset = train_subset_stratified

            target_macro_class = '0'
            num_micro = 100
            num_macro = 20
            parsed_data = []
        else:
            print(f"Dataset {dataset_name} non riconosciuto.")
            return False

    except Exception as e:
        print(f"Errore durante lo splitting: {e}")
        return False

    # 3. IDENTIFICAZIONE DELLE CLASSI DROPPATE E POOL SCARTI
    dropper = DatasetDropper(train_subset, seed=active_seed)
    train_subset_kept = dropper.drop_micro(target_macro=target_macro_class, percentage=percentage_drop / 100)
    dropped_breeds = dropper.dropped_micro_ids
    dropped_samples = [s for s in train_subset if s[1] in dropped_breeds]

    print(f"Classi non viste (droppate dal training): {len(dropped_breeds)} -> {sorted(dropped_breeds)}")

    # 4. CREAZIONE DEL SUPPORT SET FEW-SHOT
    full_support_samples, unseen_support = sample_few_shot_support(
        train_subset_kept=train_subset_kept,
        dropped_samples=dropped_samples,
        dropped_breeds=dropped_breeds,
        shots=shots,
        seed=active_seed
    )
    print(f"Campioni Support Set totali: {len(full_support_samples)} (di cui {len(unseen_support)} few-shot per classi non viste)")
    print(f"Campioni Test Set (Mondo Reale intatto): {len(test_subset)}")

    # 5. CARICAMENTO DEL MODELLO PRE-ADDESTRATO
    model = ModelsCreator(backbone_name=name, num_micro_classes=num_micro, num_macro_classes=num_macro,
                          pretrained=False).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 6. DATALOADERS PER SUPPORT E TEST SET
    if dataset_name == "pets":
        support_dataset = PetDataset(data_list=full_support_samples, transform=config.VAL_TEST_TRANSFORMS)
        test_dataset = PetDataset(data_list=test_subset, transform=config.VAL_TEST_TRANSFORMS)
    elif dataset_name == "cifar100":
        support_dataset = Cifar100Dataset(data_list=full_support_samples, image_arrays=images_train,
                                         transform=config.VAL_TEST_TRANSFORMS)
        test_dataset = Cifar100Dataset(data_list=test_subset, image_arrays=images_test,
                                       transform=config.VAL_TEST_TRANSFORMS)

    support_loader = DataLoader(
        support_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 7. CALCOLO DEI PROTOTIPI
    print("Calcolo dei prototipi di classe nello spazio di embedding...")
    prototypes = compute_prototypes(model, support_loader, num_micro, device)
    print(f"Prototipi calcolati: matrice di forma {tuple(prototypes.shape)}")

    # 8. INFERENZA SUL TEST SET (NEAREST PROTOTYPE VIA DISTANZA EUCLIDEA)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["micro_label"].to(device)

            query_features = model.backbone(images)
            query_features = torch.flatten(query_features, 1)

            # Distanza Euclidea tra ogni query e ciascun prototipo
            distances = torch.cdist(query_features, prototypes, p=2)
            preds = torch.argmin(distances, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 9. CALCOLO DELLE METRICHE
    active_labels = list(range(num_micro))
    report = classification_report(all_targets, all_preds, labels=active_labels, output_dict=True,
                                   zero_division=0.0)

    known_f1_scores = []
    unseen_f1_scores = []

    for c in range(num_micro):
        orig_id = str(c + 1) if dataset_name == "pets" else str(c)
        f1 = report[str(c)]['f1-score']
        if orig_id in dropped_breeds:
            unseen_f1_scores.append(f1)
        else:
            known_f1_scores.append(f1)

    macro_f1_globale = report['macro avg']['f1-score']
    macro_f1_note = sum(known_f1_scores) / len(known_f1_scores) if known_f1_scores else 0.0
    macro_f1_unseen = sum(unseen_f1_scores) / len(unseen_f1_scores) if unseen_f1_scores else 0.0
    accuracy = report.get('accuracy', 0.0)

    print("\n--- RISULTATI PROTO-NET (FEW-SHOT ADAPTATION) ---")
    print(f"Accuracy Globale:                  {accuracy:.4f}")
    print(f"Macro F1-Score GLOBALE:             {macro_f1_globale:.4f}")
    print(f"Macro F1-Score CLASSI NOTE:         {macro_f1_note:.4f}")
    print(f"Macro F1-Score CLASSI NON VISTE:    {macro_f1_unseen:.4f}  ({len(unseen_f1_scores)} classi a {shots}-shot)")

    # 10. MAPPATURA NOMI CLASSI PER IL CSV
    idx_to_class = {}
    if dataset_name == "pets":
        for img_path, micro_lbl, _ in parsed_data:
            filename_img = os.path.basename(str(img_path))
            breed_name = filename_img.rsplit('_', 1)[0]
            idx_to_class[str(int(micro_lbl) - 1)] = breed_name
    elif dataset_name == "cifar100":
        for i in range(100):
            idx_to_class[str(i)] = f"Cifar_Class_{i}"

    # 11. SCRITTURA CSV REPORT DETTAGLIATO PER CLASSE
    csv_classi = config.PERSISTANCE_PATH / f"report_{name}_{dataset_name}_proto_classi.csv"
    file_exists_classi = csv_classi.exists()

    with open(csv_classi, "a", newline="") as csvfile:
        fieldnames = ['drop_type', 'drop_percentage', 'seed', 'shots', 'Classe', 'Vista_In_Train',
                      'Precision', 'Recall', 'F1-score', 'Support']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists_classi:
            writer.writeheader()

        for classe_idx, metrics in report.items():
            if classe_idx in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(metrics, dict):
                nome_classe = idx_to_class.get(classe_idx, f"Classe_{classe_idx}")
                orig_id = str(int(classe_idx) + 1) if dataset_name == "pets" else str(classe_idx)
                vista = "No" if orig_id in dropped_breeds else "Si"

                writer.writerow({
                    'drop_type': typeofdrop,
                    'drop_percentage': percentage_drop,
                    'seed': active_seed,
                    'shots': shots,
                    'Classe': nome_classe,
                    'Vista_In_Train': vista,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-score': f"{metrics['f1-score']:.4f}",
                    'Support': metrics['support']
                })

    # 12. SCRITTURA CSV REPORT GLOBALE
    csv_globali = config.PERSISTANCE_PATH / f"report_{name}_{dataset_name}_proto_globali.csv"
    file_exists_glob = csv_globali.exists()

    with open(csv_globali, "a", newline="") as csvfile:
        fieldnames_glob = ['drop_type', 'drop_percentage', 'seed', 'shots', 'Accuracy',
                           'Macro_F1_Globale', 'Macro_F1_Note', 'Macro_F1_Non_Viste']
        writer_glob = csv.DictWriter(csvfile, fieldnames=fieldnames_glob)
        if not file_exists_glob:
            writer_glob.writeheader()

        writer_glob.writerow({
            'drop_type': typeofdrop,
            'drop_percentage': percentage_drop,
            'seed': active_seed,
            'shots': shots,
            'Accuracy': f"{accuracy:.4f}",
            'Macro_F1_Globale': f"{macro_f1_globale:.4f}",
            'Macro_F1_Note': f"{macro_f1_note:.4f}",
            'Macro_F1_Non_Viste': f"{macro_f1_unseen:.4f}"
        })

    print(f"Report ProtoNet salvati per {name} in {config.PERSISTANCE_PATH}\n")
    return True
