import sys
import os
import argparse
import config
import random
import numpy as np
import torch

from ModelUtility.train_model import create_and_train_model
from testmodel import TestModello
from prototypical_network import evaluate_prototypical_model

def parse_args():
    parser = argparse.ArgumentParser(description="Progetto Tesi: Micro vs Macro Drop")

    # Aggiungiamo tutti gli argomenti
    parser.add_argument('--seed', type=int, default=777, help="Seed per la riproducibilità globale")

    parser.add_argument('--dataset', type=str, choices=['pets', 'cifar100'],
                        default='cifar100', help="Dataset su cui operare")

    parser.add_argument('--op', type=str, choices=['0', '1', '2', '3'],
                        default='1', help="0: Train Completo, 1: Test Veloce, 2: Testa Salvati, 3: Prototypical Few-Shot (Classi Non Viste)")

    parser.add_argument('--shots', type=int, default=5, help="Numero di immagini di supporto per classe non vista (default: 5)")

    parser.add_argument('--model_path', type=str, default=None, help="Percorso opzionale di un singolo modello .pt da valutare con ProtoNet")

    return parser.parse_args()

def set_deterministic_env(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Backend cuDNN per determinismo hardware
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_deterministic_env(args.seed)

    dataset_name = args.dataset
    op_choice = args.op

    if op_choice == "0":
        print("Operazione: Avvia Training Completo (Baseline + Micro/Macro Drop)")
        reti = ["resnet50", "efficientnet"]
        drops = [5, 10, 15, 20, 25, 30, 35, 40]
        tipi_drop = ["macro", "micro"]

        print("\n--- INIZIO ADDESTRAMENTI BASELINE (0% Drop) ---")
        for rete in reti:
            # Ricorda di passare args.seed a questa funzione per salvare i CSV col nome giusto!
            create_and_train_model(rete, True, 0, "macro", dataset_name=dataset_name, _seed=args.seed)

        print("\n--- INIZIO ADDESTRAMENTI ESPERIMENTO (Drop Variabile) ---")
        for rete in reti:
            for drop in drops:
                for tipo in tipi_drop:
                    create_and_train_model(rete, True, drop, tipo, dataset_name=dataset_name, _seed=args.seed)

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

    elif op_choice == "3":
        print(f"Operazione: Valutazione Prototypical Networks Few-Shot ({args.shots} shots) sulle classi non viste")
        print(f"\n--- INIZIO VALUTAZIONE PROTOTYPICAL ({dataset_name.upper()}) ---")
        if args.model_path:
            if os.path.exists(args.model_path):
                evaluate_prototypical_model(args.model_path, shots=args.shots, seed=args.seed)
            else:
                print(f"File modello specificato non trovato: {args.model_path}")
        else:
            if not os.path.exists(config.PERSISTANCE_PATH):
                print("Cartella persistenza non trovata!")
                return

            modelli_testati = 0
            for file in sorted(os.listdir(config.PERSISTANCE_PATH)):
                if file.endswith(".pt") and dataset_name in file and "micro" in file:
                    percorso_modello = config.PERSISTANCE_PATH / file
                    success = evaluate_prototypical_model(percorso_modello, shots=args.shots, seed=args.seed)
                    if success:
                        modelli_testati += 1

            if modelli_testati == 0:
                print(f"Nessun modello micro-drop valido trovato per il dataset {dataset_name} in {config.PERSISTANCE_PATH}.")
            else:
                print(f"Valutazione Prototypical completata su {modelli_testati} modelli.")


if __name__ == "__main__":
    main()