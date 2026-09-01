import os
import random
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

import config
from DatasetLibrary.dataset_pytorch import PetDataset
from ModelUtility.train_model import parse_annotation_file, split_parsed_data

def crea_prototypical_loaders(K=5):
    # 1. Parsing originale
    parsed_data = parse_annotation_file()
    train_subset, val_subset, _ = split_parsed_data(
        parsed_data=parsed_data,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED
    )
    
    # 2. Raggruppa per classe usando l'indice 1 della tupla!
    elementi_per_classe = defaultdict(list)
    for item in train_subset:
        # item è (image_final_path, micro_label, macro_label)
        micro_label = item[1] 
        elementi_per_classe[micro_label].append(item)
        
    # 3. Campiona K immagini
    random.seed(config.SEED)
    support_list = []
    for label, items in elementi_per_classe.items():
        campioni = random.sample(items, min(K, len(items)))
        support_list.extend(campioni)
        
    print(f"Creato Support Set: {len(support_list)} immagini ({K} per classe).")

    # 4. Crea Dataloader
    support_dataset = PetDataset(data_list=support_list, transform=config.TRAIN_TRANSFORMS)
    val_dataset = PetDataset(data_list=val_subset, transform=config.VAL_TEST_TRANSFORMS)

    num_worker = os.cpu_count() or 0
    
    support_loader = DataLoader(support_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_worker)

    return support_loader, val_loader, train_subset

def genera_support_loader_episodico(train_subset, K=5, episodio_seed=42):
    """Genera un nuovo dataloader per il Support Set variando le 5 immagini pescate."""
    elementi_per_classe = defaultdict(list)
    for item in train_subset:
        elementi_per_classe[item[1]].append(item)
        
    random.seed(episodio_seed)
    support_list = []
    for label, items in elementi_per_classe.items():
        campioni = random.sample(items, min(K, len(items)))
        support_list.extend(campioni)
        
    support_dataset = PetDataset(data_list=support_list, transform=config.TRAIN_TRANSFORMS)
    num_worker = os.cpu_count() or 0
    return DataLoader(support_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_worker)