from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data




def Tester():
    print("Check se tutto funziona")

    # Test parser
    try:
        parsed_data = parse_annotation_file()
        print(f"Il parser funziona: Tuple estratte: {len(parsed_data)}")
    except Exception as e:
        print(f"Errore nel parser. Errore {e}")
        return


    # Test splitter
    try:
        train, val, test = split_parsed_data(parsed_data=parsed_data)
        print("Split completato con successo!")
        print(f"    - Campioni di Train:      {len(train)} (Atteso: ~5144)")
        print(f"    - Campioni di Validation: {len(val)} (Atteso: ~1102)")
        print(f"    - Campioni di Test:       {len(test)} (Atteso: ~1102)")

        print("\nVerifica struttura del primo campione del Train:")
        primo_campione = train[0]
        print(f"    - Path dell'immagine (Oggetto Path): {primo_campione[0]}")
        print(f"    - Tipo dell'oggetto Path:            {type(primo_campione[0])}")
        print(f"    - Micro-categoria (Razza):           {primo_campione[1]}")
        print(f"    - Macro-categoria (Specie):          {primo_campione[2]}")

    except Exception as e:
        print(f"Lo splitter non funziona. Errore {e}")

Tester()
