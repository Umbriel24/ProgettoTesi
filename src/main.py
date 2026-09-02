import csv
from prototypical.data_loader import crea_prototypical_loaders
from prototypical.test_prototypical import TestPrototypical
import sys
import numpy as np

from ModelUtility.train_model import create_and_train_model
from ModelUtility.train_model import check_model_existence
from prototypical.data_loader import genera_support_loader_episodico

import config
from testmodel import TestModello
from csvutility import utility_csv
def main(num: int = 10):
        
    print("Scrivi il numero per continuare l'esecuzione")
    print("1: Cerca il miglior modello tra le reti")
    print("2: Testa tutti i modelli")

    if int(num) == 1:
        utility_csv.trova_miglior_percentage()
    elif int(num) == 2:
        nets = ["resnet18", "resnet50", "densenet", "efficientnet"]
        for net in nets:
            for typeofdrop in ("micro", "macro"):
                for i in range(9):
                    model_name = f"model_{net}_percentage{i*5}_{typeofdrop}_{config.SEED}.pt"
                    if check_model_existence(model_name):
                        TestModello(config.PERSISTANCE_PATH / model_name, config.SEED)
    elif int(num) == 3:
        modelli_salvati = list(config.PERSISTANCE_PATH.glob("*.pt"))
        if not modelli_salvati:
            return
            
        # Generiamo i dati base UNA SOLA VOLTA
        _, val_loader, base_train_subset = crea_prototypical_loaders(K=5)
        NUM_EPISODI = 30 # Alzato a 30 per la run definitiva
        
        csv_path = config.PERSISTANCE_PATH / "prototypical_results_stochastic.csv"
        file_exists = csv_path.exists()
        
        # --- LOGICA DI RIPRESA (RESUME) ---
        modelli_gia_processati = set()
        if file_exists:
            with open(csv_path, mode='r') as f:
                reader = csv.reader(f)
                next(reader, None) # Salta l'intestazione
                for row in reader:
                    if row: # Sicurezza contro le righe vuote
                        modelli_gia_processati.add(row[0]) # La colonna 0 è il Nome Modello

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Nome Modello", "Episodi", "Acc Globale (Media±Std)", "Acc Note (Media±Std)", "Acc Ignote (Media±Std)", "Campioni Ignoti", "Num Classi Droppate"])

            for model_path in modelli_salvati:
                # Se è già nel Set, passa al prossimo file
                if model_path.name in modelli_gia_processati:
                    print(f" [SKIP] {model_path.name} (Già calcolato)")
                    continue
                
                print(f"\n--- Analisi Stocastica ({NUM_EPISODI} Episodi): {model_path.name} ---")
                try:
                    # Inizializza il modello (pesi caricati una sola volta)
                    tester = TestPrototypical(model_path, config.SEED, None, val_loader, base_train_subset) 
                    
                    acc_glob_list, acc_seen_list, acc_unseen_list = [], [], []
                    tot_unseen, num_dropped = 0, 0
                    
                    for ep in range(NUM_EPISODI):
                        # Varia il seed per ogni episodio aggiungendo l'indice
                        seed_episodio = config.SEED + ep
                        support_loader_ep = genera_support_loader_episodico(base_train_subset, K=5, episodio_seed=seed_episodio)
                        
                        # Esegue l'episodio
                        acc_glob, acc_seen, acc_unseen, tot_unseen, num_dropped = tester.run_episode(support_loader_ep)
                        
                        acc_glob_list.append(acc_glob)
                        acc_seen_list.append(acc_seen)
                        acc_unseen_list.append(acc_unseen)
                        
                    # Calcolo Statistico (Media e Deviazione Standard)
                    glob_mean, glob_std = np.mean(acc_glob_list), np.std(acc_glob_list)
                    seen_mean, seen_std = np.mean(acc_seen_list), np.std(acc_seen_list)
                    
                    # Le classi ignote le formattiamo solo se esistono
                    if num_dropped > 0:
                        unseen_mean, unseen_std = np.mean(acc_unseen_list), np.std(acc_unseen_list)
                        unseen_str = f"{unseen_mean:.2f} ± {unseen_std:.2f}"
                    else:
                        unseen_str = "0.00 ± 0.00"
                        
                    glob_str = f"{glob_mean:.2f} ± {glob_std:.2f}"
                    seen_str = f"{seen_mean:.2f} ± {seen_std:.2f}"

                    print(f"Risultato Finale -> Ignote: {unseen_str}% | Note: {seen_str}%")
                    writer.writerow([model_path.name, NUM_EPISODI, glob_str, seen_str, unseen_str, tot_unseen, num_dropped])
                    file.flush() 
                    
                except ValueError as e:
                    print(f" [SKIPPED] {e}")
                    continue
        modelli_salvati = list(config.PERSISTANCE_PATH.glob("*.pt"))
        if not modelli_salvati:
            return
            
        # Generiamo i dati base UNA SOLA VOLTA
        _, val_loader, base_train_subset = crea_prototypical_loaders(K=5)
        NUM_EPISODI = 30 # Inizia con 5 per testare la velocità, poi alza a 10
        
        csv_path = config.PERSISTANCE_PATH / "prototypical_results_stochastic.csv"
        file_exists = csv_path.exists()
        # --- LOGICA DI RIPRESA (RESUME) ---
        modelli_gia_processati = set()
        if file_exists:
            with open(csv_path, mode='r') as f:
                reader = csv.reader(f)
                next(reader, None) # Salta l'intestazione
                for row in reader:
                    if row: # Sicurezza contro le righe vuote
                        modelli_gia_processati.add(row[0]) # La colonna 0 è il Nome Modello

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Nome Modello", "Episodi", "Acc Globale (Media±Std)", "Acc Note (Media±Std)", "Acc Ignote (Media±Std)", "Campioni Ignoti", "Num Classi Droppate"])

            for model_path in modelli_salvati:
            # Se è già nel Set, passa al prossimo file
                if model_path.name in modelli_gia_processati:
                    print(f" [SKIP] {model_path.name} (Già calcolato)")
                    continue
                
                print(f"\n--- Analisi Stocastica ({NUM_EPISODI} Episodi): {model_path.name} ---")
                # ... il resto del tuo codice con il blocco try/except rimane identico ...
                print(f"\n--- Analisi Stocastica ({NUM_EPISODI} Episodi): {model_path.name} ---")
                try:
                    # Inizializza il modello (pesi caricati una sola volta)
                    tester = TestPrototypical(model_path, config.SEED, None, val_loader, base_train_subset) 
                    
                    acc_glob_list, acc_seen_list, acc_unseen_list = [], [], []
                    tot_unseen, num_dropped = 0, 0
                    
                    for ep in range(NUM_EPISODI):
                        # Varia il seed per ogni episodio aggiungendo l'indice
                        seed_episodio = config.SEED + ep
                        support_loader_ep = genera_support_loader_episodico(base_train_subset, K=5, episodio_seed=seed_episodio)
                        
                        # Esegue l'episodio
                        acc_glob, acc_seen, acc_unseen, tot_unseen, num_dropped = tester.run_episode(support_loader_ep)
                        
                        acc_glob_list.append(acc_glob)
                        acc_seen_list.append(acc_seen)
                        acc_unseen_list.append(acc_unseen)
                        
                    # Calcolo Statistico (Media e Deviazione Standard)
                    glob_mean, glob_std = np.mean(acc_glob_list), np.std(acc_glob_list)
                    seen_mean, seen_std = np.mean(acc_seen_list), np.std(acc_seen_list)
                    
                    # Le classi ignote le formattiamo solo se esistono
                    if num_dropped > 0:
                        unseen_mean, unseen_std = np.mean(acc_unseen_list), np.std(acc_unseen_list)
                        unseen_str = f"{unseen_mean:.2f} ± {unseen_std:.2f}"
                    else:
                        unseen_str = "0.00 ± 0.00"
                        
                    glob_str = f"{glob_mean:.2f} ± {glob_std:.2f}"
                    seen_str = f"{seen_mean:.2f} ± {seen_std:.2f}"

                    print(f"Risultato Finale -> Ignote: {unseen_str}% | Note: {seen_str}%")
                    writer.writerow([model_path.name, NUM_EPISODI, glob_str, seen_str, unseen_str, tot_unseen, num_dropped])
                    file.flush() 
                    
                except ValueError as e:
                    print(f" [SKIPPED] {e}")
                    continue
    else:
        train_from_microdrop("resnet18", 0)
        train_from_macrodrop("resnet18", 0)

        train_from_microdrop("resnet50", 0)
        train_from_macrodrop("resnet50", 0)

        train_from_microdrop("densenet", 0)
        train_from_macrodrop("densenet", 0)

        train_from_microdrop("efficientnet", 0)
        train_from_macrodrop("efficientnet", 0)



def train_from_macrodrop(subnet_name: str, percentagedrop: int):
    if percentagedrop >= 9:
        return

    # Parte da percentagedrop e arriva a 8
    for i in range(percentagedrop, 9):
        create_and_train_model(
            subnet_name,
            pre_trained_value=True,
            percentage_drop=(i * 5)
        )

def train_from_microdrop(subnet_name: str, percentagedrop: int):
    if percentagedrop >= 9:
        return
    if percentagedrop == 0:
        percentagedrop = 1

    for i in range(percentagedrop, 9):
        create_and_train_model(
            subnet_name,
            pre_trained_value=True,
            percentage_drop=(i * 5),
            typeofdrop="micro"
        )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    else:
        num = 0
    main(num)
