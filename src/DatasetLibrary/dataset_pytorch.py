from PIL import Image
from sympy.codegen.ast import none
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
    def __init__(self, data_list, transform=None, micro_mapping=None, macro_mapping=None):
        self.data_list = data_list
        self.transform = transform
        self.micro_mapping = micro_mapping
        self.macro_mapping = macro_mapping

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_final_path, micro_label, macro_label = self.data_list[index]

        try:
            image = Image.open(image_final_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"File non trovato al path {image_final_path}. Errore {e}")

        if self.transform is not None:
            image = self.transform(image)

        # Rimappatura micro classe
        if self.micro_mapping is not None:
            mapped_micro = self.micro_mapping[str(micro_label)]
        else:
            mapped_micro = int(micro_label) - 1

        # Rimappatura macro classe
        if self.macro_mapping is not None:
            mapped_macro = self.macro_mapping[str(macro_label)]
        else:
            mapped_macro = int(macro_label) - 1

        # NESSUNA SOTTRAZIONE AGGIUNTIVA QUI!
        return {
            "image": image,
            "micro_label": torch.tensor(int(mapped_micro), dtype=torch.long),
            "macro_label": torch.tensor(int(mapped_macro), dtype=torch.long)
        }