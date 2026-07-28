import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from PIL import Image

import os
import sys

# --- Rende importabili i moduli dalla cartella src ---
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DatasetLibrary.dataset_pytorch import PetDataset
from modelscreator import ModelsCreator


class TestDynamicArchitecture(unittest.TestCase):
    """
    Test suite per verificare la flessibilità dell'architettura e
    l'assenza di doppi '- 1' in dataset_pytorch.py
    """

    def setUp(self):
        self.synthetic_data = [
            ("img1.jpg", "1", "10"),
            ("img2.jpg", "3", "10"),
            ("img3.jpg", "5", "20"),
            ("img4.jpg", "1", "20"),
        ]

        self.unique_micro = sorted(list(set(x[1] for x in self.synthetic_data)))
        self.micro_mapping = {str(old_id): new_idx for new_idx, old_id in enumerate(self.unique_micro)}

        self.unique_macro = sorted(list(set(x[2] for x in self.synthetic_data)))
        self.macro_mapping = {str(old_id): new_idx for new_idx, old_id in enumerate(self.unique_macro)}

    def test_mapping_logic(self):
        self.assertEqual(len(self.micro_mapping), 3)
        self.assertEqual(self.micro_mapping["1"], 0)
        self.assertEqual(self.micro_mapping["5"], 2)

        self.assertEqual(len(self.macro_mapping), 2)
        self.assertEqual(self.macro_mapping["10"], 0)

    def test_pet_dataset_mapping(self):
        with patch("PIL.Image.open") as mock_open:
            mock_img = Image.new('RGB', (224, 224))
            mock_open.return_value = mock_img

            dataset = PetDataset(
                data_list=self.synthetic_data,
                micro_mapping=self.micro_mapping,
                macro_mapping=self.macro_mapping
            )

            sample = dataset[0]
            self.assertEqual(sample["micro_label"].item(), 0)

            sample_2 = dataset[1]
            self.assertEqual(sample_2["micro_label"].item(), 1)

    def test_models_creator_dynamic_heads(self):
        num_micro = len(self.micro_mapping)
        num_macro = len(self.macro_mapping)

        model = ModelsCreator(
            backbone_name="resnet18",
            pretrained=False,
            num_micro_classes=num_micro,
            num_macro_classes=num_macro
        )

        self.assertEqual(model.micro_head.out_features, 3)
        self.assertEqual(model.macro_head.out_features, 2)

        dummy_input = torch.randn(1, 3, 224, 224)
        out_micro, out_macro = model(dummy_input)

        self.assertEqual(out_micro.shape, (1, 3))
        self.assertEqual(out_macro.shape, (1, 2))

    def test_index_error_prevention(self):
        # Simuliamo una label alta che viene rimappata a 0
        mapping = {"99": 0}
        data = [("test.jpg", "99", "1")]

        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value = Image.new('RGB', (224, 224))
            ds = PetDataset(data, micro_mapping=mapping)
            label = ds[0]["micro_label"]

            criterion = nn.CrossEntropyLoss()
            output = torch.randn(1, 1)  # Modello con 1 sola classe
            try:
                loss = criterion(output, label.unsqueeze(0))
            except IndexError:
                self.fail("CrossEntropyLoss ha sollevato IndexError. Controlla il -1 in dataset_pytorch!")


if __name__ == "__main__":
    unittest.main()