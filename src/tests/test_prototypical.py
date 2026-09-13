import unittest
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Rende importabili i moduli dalla cartella src ---
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prototypical_network import sample_few_shot_support, compute_prototypes


class DummyDataset(Dataset):
    def __init__(self, samples):
        # samples: lista di tuple (features_tensor, micro_label)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, lbl = self.samples[idx]
        return {
            "image": feat,
            "micro_label": torch.tensor(lbl, dtype=torch.long)
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simula il backbone come identity
        self.backbone = nn.Identity()

    def forward(self, x):
        return x, x


class TestPrototypicalNetwork(unittest.TestCase):
    def setUp(self):
        # Simuliamo 3 classi: 0 (vista), 1 (non vista), 2 (non vista)
        # Classi con ID stringa come nel progetto: '0', '1', '2'
        self.seen_samples = [("path_s1.jpg", "0", "1"), ("path_s2.jpg", "0", "1"), ("path_s3.jpg", "0", "1")]
        self.dropped_samples = [
            ("path_d1.jpg", "1", "2"),
            ("path_d2.jpg", "1", "2"),
            ("path_d3.jpg", "1", "2"),
            ("path_d4.jpg", "1", "2"),
            ("path_d5.jpg", "1", "2"),
            ("path_d6.jpg", "1", "2"),
            ("path_d7.jpg", "2", "2"),
            ("path_d8.jpg", "2", "2"),
            ("path_d9.jpg", "2", "2"),
            ("path_d10.jpg", "2", "2"),
        ]
        self.dropped_breeds = {"1", "2"}

    def test_sample_few_shot_support_shots_count(self):
        shots = 3
        full_support, unseen_support = sample_few_shot_support(
            train_subset_kept=self.seen_samples,
            dropped_samples=self.dropped_samples,
            dropped_breeds=self.dropped_breeds,
            shots=shots,
            seed=777
        )

        # 3 campioni noti + (3 shots * 2 classi non viste) = 9
        self.assertEqual(len(unseen_support), 6)
        self.assertEqual(len(full_support), 9)

        # Verifica conteggio per ciascuna classe non vista
        counts_1 = sum(1 for s in unseen_support if s[1] == "1")
        counts_2 = sum(1 for s in unseen_support if s[1] == "2")
        self.assertEqual(counts_1, 3)
        self.assertEqual(counts_2, 3)

    def test_sample_few_shot_determinism(self):
        s1_full, s1_unseen = sample_few_shot_support(
            self.seen_samples, self.dropped_samples, self.dropped_breeds, shots=2, seed=42
        )
        s2_full, s2_unseen = sample_few_shot_support(
            self.seen_samples, self.dropped_samples, self.dropped_breeds, shots=2, seed=42
        )
        self.assertEqual(s1_unseen, s2_unseen)

    def test_compute_prototypes_centroid(self):
        device = torch.device("cpu")
        model = DummyModel()

        # Creiamo campioni sintetici per 2 classi nello spazio R^2:
        # Classe 0: (1.0, 1.0) e (3.0, 3.0) -> Media attesa: (2.0, 2.0)
        # Classe 1: (10.0, 0.0) e (10.0, 4.0) -> Media attesa: (10.0, 2.0)
        samples = [
            (torch.tensor([1.0, 1.0]), 0),
            (torch.tensor([3.0, 3.0]), 0),
            (torch.tensor([10.0, 0.0]), 1),
            (torch.tensor([10.0, 4.0]), 1),
        ]
        loader = DataLoader(DummyDataset(samples), batch_size=2, shuffle=False)

        prototypes = compute_prototypes(model, loader, num_classes=2, device=device)

        self.assertEqual(prototypes.shape, (2, 2))
        self.assertTrue(torch.allclose(prototypes[0], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.allclose(prototypes[1], torch.tensor([10.0, 2.0])))

    def test_nearest_prototype_classification(self):
        # Prototypes: Classe 0 a (0.0, 0.0), Classe 1 a (10.0, 10.0)
        prototypes = torch.tensor([[0.0, 0.0], [10.0, 10.0]])

        # Query vicino a Classe 0: (1.0, 0.5)
        # Query vicino a Classe 1: (9.0, 10.5)
        queries = torch.tensor([[1.0, 0.5], [9.0, 10.5]])

        dists = torch.cdist(queries, prototypes, p=2)
        preds = torch.argmin(dists, dim=1)

        self.assertEqual(preds[0].item(), 0)
        self.assertEqual(preds[1].item(), 1)


if __name__ == "__main__":
    unittest.main()
