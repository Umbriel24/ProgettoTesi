from ModelUtility.train_model import create_and_train_model
from testmodel import TestModello
from csvutility import utility_csv


if __name__ == "__main__":

    def main(num: int):
        
        print("Scrivi il numero per continuare l'esecuzione")
        print("1: Cerca il miglior modello tra le reti")
        print("2: Testa tutti i modelli")


        if int(num) == 1:
            utility_csv.trova_miglior_percentage()
        elif int(num) == 2:
            for i in range(9):
                TestModello(f"model_resnet18_percentage{i*5}.pth", 777)
        else:


            #create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=0)
            #create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=5)
            #create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=10)
            #create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=15)
            #create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=20)
            create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=25)
            create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=30)
            create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=35)
            create_and_train_model("resnet18", pre_trained_value=True, percentage_drop=40)

            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=0)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=5)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=10)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=15)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=20)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=25)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=30)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=35)
            create_and_train_model("densenet", pre_trained_value=True, percentage_drop=40)

            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=0)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=5)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=10)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=15)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=20)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=25)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=30)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=35)
            create_and_train_model("resnet50", pre_trained_value=True, percentage_drop=40)

            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=0)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=5)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=10)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=15)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=20)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=25)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=30)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=35)
            create_and_train_model("efficientnet", pre_trained_value=True, percentage_drop=40)

def train_from_macrodrop(subnet_name: str, percentagedrop: int):
    if percentagedrop >= 9:
        return

    # Parte da percentagedrop e arriva a 8
    for i in range(percentagedrop, 9):
        create_and_train_model(
            subnet_name,  # Senza virgolette
            pre_trained_value=True,
            percentage_drop=(i * 5)
        )






# sono 5 reti. Droput di macroclasse e microclasse dallo 0 al 40%. 15 per rete. Per 5 sono 75 reti
