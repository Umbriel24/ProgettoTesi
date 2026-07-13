import config

from torch.utils.data.dataloader import DataLoader
from DatasetLibrary.dataset_pytorch import PetDataset
from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data


def main():
    print("TEST PIPELINE INGESTION DATI \n")

    # 1. PARSING
    print("1. PARSING: \n")
    try:
        parsed_data = parse_annotation_file()
        print(f"PARSING COMPLETATO. Estratti {len(parsed_data)} files")
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

if __name__ == "__main__":
    main()
