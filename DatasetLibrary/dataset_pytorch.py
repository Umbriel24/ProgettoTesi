from PIL import Image
from torch.utils.data import Dataset
import torch

class PetDataset(Dataset):

    # Costruttore
    def __init__(self, data_list, base_dir, transform=None):
        self.data_list = data_list
        self.images_dir = base_dir / "images"
        self.transform = transform


    # Ritorna il numero di campioni nel subset
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        #Recupero tupla dall'indice
        image_name, micro_label, macro_label = self.data_list[index]

        # Path immagine
        img_path = self.images_dir / image_name

        # Lazy loading immagine
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"File non trovato al path {img_path}. Errore {e}")

        if (self.transform is not None):
            image = self.transform(image)

        # Converte le label in tensori long
        return {
            "image": image,
            "micro_label": torch.tensor(int(micro_label), dtype = torch.long),
            "macro_label": torch.tensor(int(macro_label), dtype = torch.long)
        }
