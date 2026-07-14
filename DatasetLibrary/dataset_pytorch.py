from PIL import Image
from torch.utils.data import Dataset
import torch

# Classe che eredita torch.utils.data.Dataset. E' un'interfaccia tra OS e modulo di training di pytorch
# Effettua:
# 1. Lazy loading dell'immagini in RAM
# 2. Estrazione micro e macro classe sottoforma di tensori
# 3. Pre-processing per il ridimensionamento a 224x224
#
#
# 1. __init_ riceve le tuple splittate contenente il path delle immagini
# 2. __len__ fornisce la grandezza (dimensione) del dataset, ovvero quanti elementi ci sono
# 3. __getitem__(indice) dato un indice, recupera l'oggetto dal disco. Lo apre con PIL, viene convertito in RGB, applica le trasformazioni e restituisce un
# dizionario per la gpu
class PetDataset(Dataset):

    # Costruttore
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform


    # Ritorna il numero di campioni nel subset
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        #Recupero tupla dall'indice
        image_final_path, micro_label, macro_label = self.data_list[index]

        # Lazy loading immagine
        try:
            image = Image.open(image_final_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"File non trovato al path {image_final_path}. Errore {e}")

        if (self.transform is not None):
            image = self.transform(image)

        # Converte le label in tensori long. Servono per le loss functions
        return {
            "image": image,
            "micro_label": torch.tensor(int(micro_label) - 1, dtype = torch.long),
            "macro_label": torch.tensor(int(macro_label) - 1, dtype = torch.long)
        }
