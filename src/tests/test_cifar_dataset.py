import unittest
import numpy as np
import torch
from PIL import Image
import os
import sys

# --- Rende importabili i moduli dalla cartella src ---
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DatasetLibrary.dataset_cifar import Cifar100Dataset


class TestCifarDataset(unittest.TestCase):
    """
    Test suite per verificare il corretto caricamento in RAM e la
    conversione vettoriale delle immagini di CIFAR-100.
    """

    def setUp(self):
        # Simuliamo 1 sola immagine (N=1).
        # CIFAR ha 3072 valori: 1024 per il Rosso, 1024 per il Verde, 1024 per il Blu.
        self.dummy_image_array = np.zeros((1, 3072), dtype=np.uint8)

        # Coloriamo di rosso puro tutti i primi 1024 valori (il canale R)
        self.dummy_image_array[0, :1024] = 255

        # Formato atteso dal Dataset: (Indice_RAM, Micro_str, Macro_str)
        self.dummy_data_list = [(0, "5", "10")]

    def test_numpy_to_pil_reshape(self):
        """
        Verifica che l'array 1D (3072) venga correttamente rimodellato in una PIL Image
        32x32 con i canali colore nel giusto ordine (RGB).
        """
        dataset = Cifar100Dataset(data_list=self.dummy_data_list, image_arrays=self.dummy_image_array)

        sample = dataset[0]
        img = sample["image"]

        # 1. Verifica il tipo di output
        self.assertIsInstance(img, Image.Image, "L'output deve essere un'immagine PIL")

        # 2. Verifica le dimensioni spaziali
        self.assertEqual(img.size, (32, 32), "Le dimensioni devono essere 32x32 pixel")

        # 3. Verifica l'allineamento dei canali (Il primo pixel deve essere RGB: 255, 0, 0)
        r, g, b = img.getpixel((0, 0))
        self.assertEqual(r, 255, "Il canale Rosso è allineato male!")
        self.assertEqual(g, 0, "Il canale Verde è sporco!")
        self.assertEqual(b, 0, "Il canale Blu è sporco!")

    def test_label_tensor_conversion(self):
        """
        Verifica che le label testuali estratte dal parser vengano correttamente
        convertite in tensori PyTorch Long, pronti per la CrossEntropy.
        """
        dataset = Cifar100Dataset(data_list=self.dummy_data_list, image_arrays=self.dummy_image_array)

        sample = dataset[0]
        micro = sample["micro_label"]
        macro = sample["macro_label"]

        self.assertIsInstance(micro, torch.Tensor)
        self.assertEqual(micro.dtype, torch.long)
        self.assertEqual(micro.item(), 5)  # La micro era "5"

        self.assertIsInstance(macro, torch.Tensor)
        self.assertEqual(macro.item(), 10)  # La macro era "10"

if __name__ == "__main__":
    unittest.main()