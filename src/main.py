from ModelUtility.train import CreateAndRunTrainingModel


if __name__ == "__main__":
    #CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=0)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=5)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=10)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=15)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=20)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=25)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=30)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=35)
    CreateAndRunTrainingModel("resnet18", pre_trained_value=True, percentageDrop=40)




# sono 5 reti. Droput di macroclasse e microclasse dallo 0 al 40%. 15 per rete. Per 5 sono 75 reti
