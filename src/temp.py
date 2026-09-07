import os
import glob
import pandas as pd
from pathlib import Path


def crea_csv_unito_global(path_directory):
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

def crea_csv_unito_classi(path_directory):
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
    pattern = os.path.join(path_directory, "report_*_classi.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Nessun file trovato con il pattern 'report_*_classi.csv' in '{path_directory}'")
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
            if filename.startswith("report_") and filename.endswith("_classi"):
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
    output_file = os.path.join(path_directory, "report_tutte_reti_classi_unito.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nFile unito salvato come: {output_file}")
    print(f"Totale righe: {len(merged_df)}")
    print(f"Colonne: {list(merged_df.columns)}")
    print("\nNetwork trovate:")
    for net in merged_df['network'].unique():
        count = len(merged_df[merged_df['network'] == net])
        print(f"  - {net}: {count} righe")
    
    return merged_df

if __name__ == "__main__":
    crea_csv_unito_classi("/home/umbriel24/Documenti/Tesi/")