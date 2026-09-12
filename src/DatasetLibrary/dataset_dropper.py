import random
import config
from collections import defaultdict

class DatasetDropper:
    def __init__(self, data_list: list, seed: int = config.SEED):
        self.data_list = list(data_list)
        self.seed = seed
        # Razze (microclassi) rimosse INTERAMENTE dall'ultimo drop_micro.
        # Serve per applicare lo stesso drop a validation/test (Opzione 1).
        self.dropped_micro_ids = set()


    # Rimuove una percentuale di campioni da macro
    def drop_macro(self, target_macro: str, percentage: float) -> list:

        if not (0.0 <= percentage <= 1.0):
            raise ValueError("Errore, la percentuale non è compresa tra 0 e 1")

        rng = random.Random(self.seed)

        # separazione target cani e gatti
        target_samples = [x for x in self.data_list if str(x[2]) == str(target_macro)]
        other_samples = [x for x in self.data_list if str(x[2]) != str(target_macro)]

        # Raggruppamento microclasse
        micro_groups = defaultdict(list)
        for sample in target_samples:
            micro_groups[sample[1]].append(sample)

        kept_target_samples = []

        # Applichiamo il drop su ogni microclasse
        for micro_id, samples in micro_groups.items():
            # Copiamo e mescoliamo localmente per riproducibilità
            local_samples = list(samples)
            rng.shuffle(local_samples)

            num_to_drop = round(len(local_samples) * percentage)
            num_to_keep = max(0, len(local_samples) - num_to_drop)

            kept_target_samples.extend(local_samples[:num_to_keep])

        return other_samples + kept_target_samples

    def drop_micro(self, target_macro: str, percentage: float) -> list:
        """
        Rimuove intere microclassi dal dataset.
        """

        if not (0.0 <= percentage <= 1.0):
            raise ValueError("Errore, la percentuale non è compresa tra 0 e 1")

        rng = random.Random(self.seed)

        self.dropped_micro_ids = set()

        micro_groups = defaultdict(list)

        for sample in self.data_list:
            micro_groups[sample[1]].append(sample)

        micro_keys = sorted(micro_groups.keys())

        num_classes_to_drop = round(len(micro_keys) * percentage)

        rng.shuffle(micro_keys)

        classes_to_drop = set(micro_keys[:num_classes_to_drop])

        self.dropped_micro_ids = classes_to_drop

        kept_samples = [
            sample
            for sample in self.data_list
            if sample[1] not in classes_to_drop
        ]

        return kept_samples

    @staticmethod
    def remove_micro_classes(
        data_list: list,
        micro_ids_to_remove,
        target_macro: str = None
    ) -> list:
        """
        Rimuove da data_list tutti i campioni appartenenti alle
        microclassi indicate, indipendentemente dalla macroclasse.
        """

        if not micro_ids_to_remove:
            return list(data_list)

        ids = set(micro_ids_to_remove)

        return [
            x for x in data_list
            if x[1] not in ids
        ]
