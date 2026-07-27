import os
import sys
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from PIL import Image

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DatasetLibrary.dataset_pytorch import PetDataset
from modelscreator import ModelsCreator

class MyTestCase(unittest.TestCase):
    """
    Test suite per verificare la flessibilità dell'architettura
    rispetto al numero dinamico di classi (Micro e Macro).
    """

    def setUp(self):
        # Formato: (path, micro_id, macro_id)
        # Notare i "salti" negli ID (es. manca il 2 e il 4) per testare il mapping
        self.synthetic_data = [
            ("img1.jpg", "1", "10"),
            ("img2.jpg", "3", "10"),
            ("img3.jpg", "5", "20"),
            ("img4.jpg", "1", "20"),
        ]

        # Logica che verrebbe usata in train_model.py
        self.unique_micro = sorted(list(set(x[1] for x in self.synthetic_data)))
        self.micro_mapping = {old_id: new_idx for new_idx, old_id in enumerate(self.unique_micro)}

        self.unique_macro = sorted(list(set(x[2] for x in self.synthetic_data)))
        self.macro_mapping = {old_id: new_idx for new_idx, old_id in enumerate(self.unique_macro)}

    def test_mapping_logic(self):
        # Micro: "1", "3", "5" -> devono diventare 0, 1, 2
        self.assertEqual(len(self.micro_mapping), 3)
        self.assertEqual(self.micro_mapping["1"], 0)
        self.assertEqual(self.micro_mapping["5"], 2)

        # Macro: "10", "20" -> devono diventare 0, 1
        self.assertEqual(len(self.macro_mapping), 2)
        self.assertEqual(self.macro_mapping["10"], 0)

    def test_pet_dataset_mapping(self):
        # Mocking di Image.open per evitare caricamento file reali
        with patch("PIL.Image.open") as mock_open:
            mock_img = Image.new('RGB', (224, 224))
            mock_open.return_value = mock_img

            # Passiamo il mapping al dataset (come suggerito per il punto 4)
            dataset = PetDataset(
                data_list=self.synthetic_data,
                micro_mapping=self.micro_mapping
                # Qui servirebbe anche macro_mapping se implementato come discusso
            )

            # Recuperiamo il primo elemento (micro_id "1" -> deve restituire 0)
            sample = dataset[0]
            self.assertEqual(sample["micro_label"].item(), 0)

            # Recuperiamo il secondo elemento (micro_id "3" -> deve restituire 1)
            sample_2 = dataset[1]
            self.assertEqual(sample_2["micro_label"].item(), 1)

    def test_models_creator_dynamic_heads(self):
        num_micro = len(self.micro_mapping)
        num_macro = len(self.macro_mapping)

        # Istanziamo il modello con i parametri dinamici
        model = ModelsCreator(
            backbone_name="resnet18",
            pretrained=False,
            num_micro_classes=num_micro
        )

        # Verifica dimensioni Micro Head
        self.assertEqual(model.micro_head.out_features, 3)

        dummy_input = torch.randn(1, 3, 224, 224)
        out_micro, out_macro = model(dummy_input)

        self.assertEqual(out_micro.shape, (1, 3))  # 3 classi micro
        # Se hai reso dinamico anche il macro:
        # self.assertEqual(out_macro.shape, (1, 2))

    def test_index_error_prevention(self):
        # Simuliamo una label originale alta "99" che viene rimappata a 0
        mapping = {"99": 0}
        data = [("test.jpg", "99", "1")]

        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value = Image.new('RGB', (224, 224))
            ds = PetDataset(data, micro_mapping=mapping)
            label = ds[0]["micro_label"]

            # Se la label è 0, la CrossEntropyLoss non esploderà con un modello a 1 output
            criterion = nn.CrossEntropyLoss()
            output = torch.randn(1, 1)  # Modello con 1 sola classe
            try:
                loss = criterion(output, label.unsqueeze(0))
            except IndexError:
                self.fail("CrossEntropyLoss ha sollevato IndexError nonostante il mapping!")


if __name__ == '__main__':
    unittest.main()
