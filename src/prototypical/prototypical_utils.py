import random
from collections import defaultdict
from torch.utils.data import DataLoader
# Assicurati di importare PetDataset, parse_annotation_file, split_parsed_data

def crea_prototypical_loaders(K=5):
    """
    Estrae il Support Set (K immagini per classe, comprese le droppate)
    e restituisce anche il loader di validazione (Query Set).
    """
    # 1. Parsing originale (dati puliti)
    parsed_data = parse_annotation_file()
    train_subset, val_subset, _ = split_parsed_data(
        parsed_data=parsed_data,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED
    )
    
    # 2. Raggruppa i dati di training per classe micro
    elementi_per_classe = defaultdict(list)
    for item in train_subset:
        # NOTA: adatta 'micro_label' alla chiave o indice corretto del tuo dizionario/tupla
        label = item['micro_label'] Se il tuo parsing usa chiavi diverse, cambiala.
        elementi_per_classe[label].append(item)
        
    # 3. Campiona casualmente K immagini per ogni classe
    random.seed(config.SEED) # Riproducibilità per la tesi!
    support_list = []
    for label, items in elementi_per_classe.items():
        campioni = random.sample(items, min(K, len(items)))
        support_list.extend(campioni)
        
    print(f"Creato Support Set con {len(support_list)} immagini totali ({K} per classe).")

    # 4. Crea Dataset e Dataloader usando le classi che hai già
    support_dataset = PetDataset(data_list=support_list, transform=config.TRAIN_TRANSFORMS)
    val_dataset = PetDataset(data_list=val_subset, transform=config.VAL_TEST_TRANSFORMS)

    num_worker = os.cpu_count() or 0
    support_loader = DataLoader(support_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_worker)

    return support_loader, val_loader