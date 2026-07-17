import csv
from config import BASE_DIR

def trova_miglior_percentage():

    miglior_modello = ""

    for i in range(9):
        with open(BASE_DIR / "src" /  f"resnet18_percentage{i*5}.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)
            best_value = 100
            for line in reader:
                temp_val_loss = line['val_loss']
                line = float(temp_val_loss)

                if line < best_value:
                    best_value = line
                    miglior_modello = f"Il miglior modello è resnet18_percentage{i*5}.csv, con percentuale = {best_value}"

    print(miglior_modello)
