import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Funziona solo per leggere i file di Kaggle -> Cifar-100
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


def parse_cifar100_kaggle(data_pre_path):
    """
    Legge i file pickle di CIFAR-100 e restituisce due tuple (train e test).
    Ogni tupla contiene: (lista_dati_per_dropper, array_immagini)
    """
    data_train_path = f"{data_pre_path}/train"
    data_test_path = f"{data_pre_path}/test"

    # Lettura dizionari
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)

    # Estrazione array numpy delle immagini (N, 3072)
    images_train = data_train_dict[b'data']
    images_test = data_test_dict[b'data']

    # Estrazione etichette:
    # fine_labels = 100 classi (Micro)
    # coarse_labels = 20 superclassi (Macro)
    micro_train = data_train_dict[b'fine_labels']
    macro_train = data_train_dict[b'coarse_labels']

    micro_test = data_test_dict[b'fine_labels']
    macro_test = data_test_dict[b'coarse_labels']

    # Creiamo la lista compatibile con il nostro DatasetDropper
    # Formato: (Indice_RAM, Micro_str, Macro_str)
    # Convertiamo in stringa per mantenere la retrocompatibilità col Dropper
    parsed_train = []
    for i in range(len(micro_train)):
        parsed_train.append((i, str(micro_train[i]), str(macro_train[i])))

    parsed_test = []
    for i in range(len(micro_test)):
        parsed_test.append((i, str(micro_test[i]), str(macro_test[i])))

    return (parsed_train, images_train), (parsed_test, images_test)


class Cifar100Dataset(Dataset):
    def __init__(self, data_list, image_arrays, transform=None):
        """
        data_list: La lista degli elementi sopravvissuti al drop [(indice, micro, macro), ...]
        image_arrays: L'array Numpy gigante con tutte le immagini originali
        """
        self.data_list = data_list
        self.image_arrays = image_arrays
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_idx, micro_label, macro_label = self.data_list[index]

        # Recupera l'array di 3072 valori corrispondente all'immagine
        img_flat = self.image_arrays[int(img_idx)]

        # Magia di Numpy: Reshape a 3 canali, 32x32 pixel e trasposizione per PIL
        img_reshaped = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)

        # Creazione dell'immagine vera e propria
        image = Image.fromarray(img_reshaped)

        if self.transform is not None:
            image = self.transform(image)

        # Nelle label non facciamo il -1 perché in CIFAR partono già da 0!
        return {
            "image": image,
            "micro_label": torch.tensor(int(micro_label), dtype=torch.long),
            "macro_label": torch.tensor(int(macro_label), dtype=torch.long)
        }