import config
import csv
import sys
import numpy as np
import pandas as pd
import os
import glob

from pathlib import Path
from ModelUtility.train_model import create_and_train_model
from ModelUtility.train_model import check_model_existence
from prototypical.data_loader import genera_support_loader_episodico
from prototypical.data_loader import crea_prototypical_loaders
from prototypical.test_prototypical import TestPrototypical
from testmodel import TestModello
from csvutility import utility_csv



def main(num: int = 0):
        
    print("Scrivi il numero per continuare l'esecuzione")
    print("1: Cerca il miglior modello tra le reti")
    print("2: Testa tutti i modelli")

    if int(num) == 0:
        return
    elif int(num) == 1:
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
    
    elif int(num) == 4:
        print("\n=== INDAGINE QUALITATIVA SUL CROLLO DEL 20% ===")
        _, val_loader, base_train_subset = crea_prototypical_loaders(K=5)
        # Costruiamo il dizionario dinamico {ID_Classe: "Nome_Razza"} dal file path
        id_to_name = {}
        for item in base_train_subset:
            percorso_file = str(item[0]) # es: "/path/dataset/Abyssinian_12.jpg"
            micro_label = item[1]        # es: 0
            
            if micro_label not in id_to_name:
                # Estraiamo "Abyssinian_12.jpg"
                nome_file = percorso_file.split('/')[-1] 
                # Separiamo all'ultimo underscore per togliere il numero: "Abyssinian"
                nome_razza = nome_file.rsplit('_', 1)[0] 
                id_to_name[micro_label] = nome_razza

        modelli_target = [
            "model_efficientnet_percentage15_micro_777.pt",
            "model_efficientnet_percentage20_micro_777.pt",
            "model_efficientnet_percentage25_micro_777.pt"
        ]

        for nome_file in modelli_target:
            path = config.PERSISTANCE_PATH / nome_file
            if path.exists():
                tester = TestPrototypical(path, config.SEED, None, val_loader, base_train_subset)
                
                # --- INIZIO BLOCCO BULLETPROOF ---
                id_to_name = {}
                for item in base_train_subset:
                    percorso_file = str(item[0])
                    micro_label = int(item[1]) # FORZIAMO A INTERO!
                    
                    if micro_label not in id_to_name:
                        nome_file = percorso_file.split('/')[-1] 
                        nome_razza = nome_file.rsplit('_', 1)[0] 
                        id_to_name[micro_label] = nome_razza
                
                id_droppati = tester.dropped_classes 
                
                # Usiamo .get() che è sicuro contro i KeyError, e forziamo 'i' a intero
                razze_escluse = [id_to_name.get(int(i), f"ID non trovato ({i})") for i in id_droppati]
                # --- FINE BLOCCO BULLETPROOF ---

                print(f"\n{nome_file} ({len(id_droppati)} classi droppate):")
                for razza in razze_escluse:
                    print(f" - {razza}")
            else:
                print(f"[ERRORE] File mancante: {nome_file}")
    
    elif int(num) == 5:
        crea_csv_unito("/home/umbriel24/Documenti/Tesi/")
    else:
        nets = ["resnet18", "resnet50", "densenet", "efficientnet"]
        for net in nets:
            train_from_macrodrop(net, 0)
            train_from_microdrop(net, 0)


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



def crea_csv_unito(path_directory):
    """
    Unisce tutti i CSV report_*_globali.csv delle diverse reti in un unico file.
    
    Args:
        path_directory (str): Path della directory contenente i file CSV
    
    Returns:
        pandas.DataFrame: DataFrame unito con tutti i dati
    """
    
    # Verifica che la directory esista
    if not os.path.exists(path_directory):
        print(f"✗ Errore: La directory '{path_directory}' non esiste!")
        return None
    
    # Costruisci il pattern di ricerca
    pattern = os.path.join(path_directory, "report_*_globali.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Nessun file trovato con il pattern 'report_*_globali.csv' in '{path_directory}'")
        return None
    
    print(f"Trovati {len(files)} file in '{path_directory}':")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    all_dataframes = []
    
    for file in files:
        try:
            # Leggi il CSV
            df = pd.read_csv(file)
            
            # Estrai il nome della rete dal filename
            filename = Path(file).stem  # toglie l'estensione .csv
            
            # Estrai la parte tra "report_" e "_globali"
            if filename.startswith("report_") and filename.endswith("_globali"):
                # Togli "report_" dall'inizio
                temp = filename[7:]  # len("report_") = 7
                # Togli "_globali" dalla fine
                network_name = temp[:-8]  # len("_globali") = 8
            else:
                # Fallback: usa il filename completo
                network_name = filename
            
            # Aggiungi la colonna con il nome della rete
            df['network'] = network_name
            
            all_dataframes.append(df)
            print(f"  ✓ {os.path.basename(file)}: {len(df)} righe, network='{network_name}'")
            
        except Exception as e:
            print(f"  ✗ Errore nel leggere {os.path.basename(file)}: {e}")
    
    if not all_dataframes:
        print("Nessun file processato correttamente!")
        return None
    
    # Unisci tutti i dataframe
    print("\nUnione dei file...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Riordina le colonne per mettere 'network' all'inizio
    cols = ['network'] + [col for col in merged_df.columns if col != 'network']
    merged_df = merged_df[cols]
    
    # Salva il file unito nella stessa directory
    output_file = os.path.join(path_directory, "report_tutte_reti_globali_unito.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nFile unito salvato come: {output_file}")
    print(f"Totale righe: {len(merged_df)}")
    print(f"Colonne: {list(merged_df.columns)}")
    print("\nNetwork trovate:")
    for net in merged_df['network'].unique():
        count = len(merged_df[merged_df['network'] == net])
        print(f"  - {net}: {count} righe")
    
    return merged_df
