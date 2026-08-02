import sys

from ModelUtility.train_model import create_and_train_model
from ModelUtility.train_model import check_model_existence

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
        net = "mlp"
        for typeofdrop in ("micro", "macro"):
            for i in range(9):
                model_name = f"model_{net}_percentage{i*5}_{typeofdrop}.pt"
                if check_model_existence(model_name):
                    TestModello(config.PERSISTANCE_PATH / model_name, config.SEED)
    else:
        train_from_microdrop("resnet18", 0)
        train_from_macrodrop("resnet18", 0)

        train_from_microdrop("resnet50", 0)
        train_from_macrodrop("resnet50", 0)

        train_from_microdrop("densenet", 0)
        train_from_macrodrop("densenet", 0)

        train_from_microdrop("efficientnet", 0)
        train_from_macrodrop("efficientnet", 0)

        train_from_microdrop("mlp", 0)
        train_from_macrodrop("mlp", 0)


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
