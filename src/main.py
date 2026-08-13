import sys
import os
import config
from ModelUtility.train_model import create_and_train_model
from testmodel import TestModello


def main():
    print("=== PROGETTO TESI: MICRO VS MACRO DROP ===")
    op_choice = "0"

    if op_choice == "0":
        reti = ["resnet18", "resnet50", "densenet", "efficientnet"]
        drops = [5, 10, 15, 20, 25, 30, 35, 40]
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
        print("\n--- TEST VELOCE MLP ---")
        create_and_train_model("mlp", False, 5, "micro", dataset_name=dataset_name)

    elif op_choice == "2":
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
    # Se passo un argomento da riga di comando (es. !python main.py 0) mantengo la retrocompatibilità
    if len(sys.argv) > 1:
        print("Usa l'interfaccia interattiva senza argomenti: !python main.py")
    else:
        main()