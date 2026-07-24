"""
Unit test per DatasetLibrary.dataset_dropper.DatasetDropper.

Verifica il comportamento di:
  - drop_macro  : drop STRATIFICATO (rimuove una % da OGNI razza del macro target)
  - drop_micro  : drop NON stratificato (rimuove INTERE razze fino a un budget)
  - remove_micro_classes : helper che replica il drop di razze su val/test (Opzione 1)

I test lavorano su liste sintetiche (path, micro_label, macro_label) e NON
richiedono torchvision né il dataset: il modulo `config` viene sostituito da
uno stub prima dell'import, così `python -m unittest` gira ovunque.

Esecuzione:
    cd src && python -m unittest tests.test_dataset_dropper -v
"""

import os
import sys
import types
import unittest

# --- Rende importabili sia `config` che `DatasetLibrary` dalla cartella src ---
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Stub di `config`: dataset_dropper fa `import config` e usa config.SEED ---
if "config" not in sys.modules:
    _stub = types.ModuleType("config")
    _stub.SEED = 777
    sys.modules["config"] = _stub

from DatasetLibrary.dataset_dropper import DatasetDropper  # noqa: E402

# Macroclasse target del progetto: '2' = cani. Tutto il resto = altri (gatti).
DOG = "2"
CAT = "1"


def make_samples(macro, breed, n, start=0):
    """Crea n tuple (path, micro_label, macro_label) per una data razza."""
    return [(f"{macro}_{breed}_{i}.jpg", breed, macro) for i in range(start, start + n)]


def dogs(data):
    return [x for x in data if str(x[2]) == DOG]


def cats(data):
    return [x for x in data if str(x[2]) == CAT]


def breed_counts(data):
    from collections import Counter
    return Counter(x[1] for x in data)


class TestDropMacro(unittest.TestCase):
    """drop_macro: riduzione proporzionale su ogni razza del macro target."""

    def setUp(self):
        # 2 razze di cani da 10 campioni + 1 razza di gatti da 10.
        self.data = (
            make_samples(DOG, "beagle", 10)
            + make_samples(DOG, "boxer", 10)
            + make_samples(CAT, "tabby", 10)
        )

    def test_drop_zero_keeps_everything(self):
        kept = DatasetDropper(self.data, seed=777).drop_macro(DOG, 0.0)
        self.assertEqual(len(kept), len(self.data))
        self.assertEqual(len(dogs(kept)), 20)

    def test_drop_all_removes_only_target_macro(self):
        kept = DatasetDropper(self.data, seed=777).drop_macro(DOG, 1.0)
        self.assertEqual(len(dogs(kept)), 0)
        self.assertEqual(len(cats(kept)), 10)  # i gatti restano intatti

    def test_stratified_removal_per_breed(self):
        # 50% -> round(10*0.5)=5 rimossi da OGNI razza -> 5 tenuti per razza.
        kept = DatasetDropper(self.data, seed=777).drop_macro(DOG, 0.5)
        counts = breed_counts(dogs(kept))
        self.assertEqual(counts["beagle"], 5)
        self.assertEqual(counts["boxer"], 5)
        self.assertEqual(len(cats(kept)), 10)

    def test_cats_never_touched(self):
        kept = DatasetDropper(self.data, seed=777).drop_macro(DOG, 0.3)
        self.assertEqual(sorted(cats(kept)), sorted(cats(self.data)))

    def test_deterministic_with_same_seed(self):
        a = DatasetDropper(self.data, seed=777).drop_macro(DOG, 0.5)
        b = DatasetDropper(self.data, seed=777).drop_macro(DOG, 0.5)
        self.assertEqual(sorted(a), sorted(b))

    def test_invalid_percentage_raises(self):
        with self.assertRaises(ValueError):
            DatasetDropper(self.data, seed=777).drop_macro(DOG, 1.5)
        with self.assertRaises(ValueError):
            DatasetDropper(self.data, seed=777).drop_macro(DOG, -0.1)


class TestDropMicro(unittest.TestCase):
    """drop_micro: rimuove INTERE razze fino a esaurire il budget di campioni."""

    def setUp(self):
        # Cani: 4 razze di dimensioni diverse (10+20+30+40 = 100) + 15 gatti.
        self.data = (
            make_samples(DOG, "A", 10)
            + make_samples(DOG, "B", 20)
            + make_samples(DOG, "C", 30)
            + make_samples(DOG, "D", 40)
            + make_samples(CAT, "X", 15)
        )
        self.total_dogs = 100

    def test_drop_zero_keeps_everything(self):
        d = DatasetDropper(self.data, seed=777)
        kept = d.drop_micro(DOG, 0.0)
        self.assertEqual(len(kept), len(self.data))
        self.assertEqual(d.dropped_micro_ids, set())

    def test_drop_all_removes_every_breed(self):
        d = DatasetDropper(self.data, seed=777)
        kept = d.drop_micro(DOG, 1.0)
        self.assertEqual(len(dogs(kept)), 0)
        self.assertEqual(len(cats(kept)), 15)
        self.assertEqual(d.dropped_micro_ids, {"A", "B", "C", "D"})

    def test_budget_exactly_respected(self):
        # 30% di 100 = 30 campioni da rimuovere -> 70 cani tenuti.
        d = DatasetDropper(self.data, seed=777)
        kept = d.drop_micro(DOG, 0.30)
        self.assertEqual(len(dogs(kept)), 70)

    def test_dropped_breeds_are_removed_whole(self):
        # Le razze in dropped_micro_ids devono sparire COMPLETAMENTE dal risultato.
        d = DatasetDropper(self.data, seed=777)
        kept = d.drop_micro(DOG, 0.30)
        remaining = breed_counts(dogs(kept))
        for breed in d.dropped_micro_ids:
            self.assertNotIn(breed, remaining, f"razza {breed} dovrebbe essere assente")

    def test_at_most_one_partial_breed(self):
        # Strategia B: solo una razza "di confine" può essere tagliata parzialmente;
        # tutte le altre razze rimaste devono avere la loro dimensione ORIGINALE.
        d = DatasetDropper(self.data, seed=777)
        kept = d.drop_micro(DOG, 0.30)
        original = {"A": 10, "B": 20, "C": 30, "D": 40}
        remaining = breed_counts(dogs(kept))
        partial = [b for b, c in remaining.items() if c != original[b]]
        self.assertLessEqual(len(partial), 1)
        # La razza parziale NON deve comparire tra quelle "eliminate del tutto".
        for b in partial:
            self.assertNotIn(b, d.dropped_micro_ids)

    def test_cats_never_touched(self):
        d = DatasetDropper(self.data, seed=777)
        kept = d.drop_micro(DOG, 0.5)
        self.assertEqual(sorted(cats(kept)), sorted(cats(self.data)))

    def test_deterministic_with_same_seed(self):
        a = DatasetDropper(self.data, seed=777)
        ra = a.drop_micro(DOG, 0.4)
        b = DatasetDropper(self.data, seed=777)
        rb = b.drop_micro(DOG, 0.4)
        self.assertEqual(sorted(ra), sorted(rb))
        self.assertEqual(a.dropped_micro_ids, b.dropped_micro_ids)

    def test_dropped_ids_reset_between_calls(self):
        # Rieseguire il drop non deve accumulare le razze del run precedente.
        d = DatasetDropper(self.data, seed=777)
        d.drop_micro(DOG, 1.0)
        self.assertEqual(d.dropped_micro_ids, {"A", "B", "C", "D"})
        d.drop_micro(DOG, 0.0)
        self.assertEqual(d.dropped_micro_ids, set())

    def test_invalid_percentage_raises(self):
        with self.assertRaises(ValueError):
            DatasetDropper(self.data, seed=777).drop_micro(DOG, 2.0)
        with self.assertRaises(ValueError):
            DatasetDropper(self.data, seed=777).drop_micro(DOG, -0.5)


class TestRemoveMicroClasses(unittest.TestCase):
    """remove_micro_classes: applica a val/test lo stesso drop di razze (Opzione 1)."""

    def setUp(self):
        self.data = (
            make_samples(DOG, "beagle", 5)
            + make_samples(DOG, "boxer", 5)
            + make_samples(CAT, "tabby", 5)
        )

    def test_removes_only_listed_breeds(self):
        out = DatasetDropper.remove_micro_classes(self.data, {"beagle"}, DOG)
        counts = breed_counts(out)
        self.assertNotIn("beagle", counts)
        self.assertEqual(counts["boxer"], 5)
        self.assertEqual(counts["tabby"], 5)

    def test_only_within_target_macro(self):
        # Se una razza con lo stesso id esiste in un altro macro, NON va toccata.
        data = self.data + make_samples(CAT, "beagle", 3)  # id "beagle" ma è gatto
        out = DatasetDropper.remove_micro_classes(data, {"beagle"}, DOG)
        cat_beagles = [x for x in out if x[1] == "beagle" and str(x[2]) == CAT]
        dog_beagles = [x for x in out if x[1] == "beagle" and str(x[2]) == DOG]
        self.assertEqual(len(cat_beagles), 3)  # gatto "beagle" preservato
        self.assertEqual(len(dog_beagles), 0)  # cane beagle rimosso

    def test_empty_ids_returns_copy_unchanged(self):
        out = DatasetDropper.remove_micro_classes(self.data, set(), DOG)
        self.assertEqual(sorted(out), sorted(self.data))
        self.assertIsNot(out, self.data)  # è una copia, non lo stesso oggetto

    def test_matches_drop_micro_selection(self):
        # Integrazione: le razze eliminate dal training spariscono anche da "val".
        train = (
            make_samples(DOG, "A", 10)
            + make_samples(DOG, "B", 20)
            + make_samples(DOG, "C", 30)
            + make_samples(CAT, "X", 10)
        )
        val = (
            make_samples(DOG, "A", 3)
            + make_samples(DOG, "B", 3)
            + make_samples(DOG, "C", 3)
            + make_samples(CAT, "X", 3)
        )
        d = DatasetDropper(train, seed=777)
        d.drop_micro(DOG, 1.0)  # rimuove tutte le razze di cani
        val_out = DatasetDropper.remove_micro_classes(val, d.dropped_micro_ids, DOG)
        self.assertEqual(len(dogs(val_out)), 0)
        self.assertEqual(len(cats(val_out)), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
