import sys
import os
import argparse
import config
from ModelUtility.train_model import create_and_train_model
from testmodel import TestModello


def main():
    # Imposta il parser degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Progetto Tesi: Micro vs Macro Drop")

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['pets', 'cifar100'],
        default='cifar100',
        help="Scegli il dataset su cui operare (pets o cifar100). Default: cifar100"
    )

    parser.add_argument(
        '--op',
        type=str,
        choices=['0', '1', '2'],
        default='1',
        help="0: Train Completo, 1: Test Veloce MLP, 2: Testa Modelli Salvati. Default: 1"
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    op_choice = args.op

    print("=== PROGETTO TESI: MICRO VS MACRO DROP ===")
    print(f"Dataset selezionato: {dataset_name.upper()}")

    if op_choice == "0":
        print("Operazione: Avvia Training Completo (Baseline + Micro/Macro Drop)")
        reti = ["resnet18", "resnet50", "densenet", "efficientnet"]
        drops = [5, 10, 15, 20]
        tipi_drop = ["macro", "micro"]

        print("\n--- INIZIO ADDESTRAMENTI BASELINE (0% Drop) ---")
        for rete in reti:
            create_and_train_model(rete, True, 0, "macro", dataset_name=dataset_name)

        print("\n--- INIZIO ADDESTRAMENTI ESPERIMENTO (Drop Variabile) ---")
        for rete in reti:
            for drop in drops:
                for tipo in tipi_drop:
                    create_and_train_model(rete, True, drop, tipo, dataset_name=dataset_name)

    elif op_choice == "1":
        print("Operazione: Avvia Training Singolo Veloce (Solo MLP 5% Micro - Test)")
        print("\n--- TEST VELOCE MLP ---")
        create_and_train_model("mlp", False, 5, "micro", dataset_name=dataset_name)

    elif op_choice == "2":
        print("Operazione: Testa tutti i modelli salvati per questo dataset")
        print(f"\n--- INIZIO TEST MODELLI ({dataset_name.upper()}) ---")
        modelli_testati = 0
        if not os.path.exists(config.PERSISTANCE_PATH):
            print("Cartella persistenza non trovata!")
            return

        for file in os.listdir(config.PERSISTANCE_PATH):
            # Filtra solo i file .pt che appartengono al dataset scelto
            if file.endswith(".pt") and dataset_name in file:
                percorso_modello = config.PERSISTANCE_PATH / file
                TestModello(percorso_modello)
                modelli_testati += 1

        if modelli_testati == 0:
            print(f"Nessun modello trovato per il dataset {dataset_name}.")
        else:
            print(f"Test completato su {modelli_testati} modelli.")


if __name__ == "__main__":
    main()