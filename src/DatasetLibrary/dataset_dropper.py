import random
import config
from collections import defaultdict

class DatasetDropper:
    def __init__(self, data_list: list, seed: int = config.SEED):
        self.data_list = list(data_list)
        self.seed = seed


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
            STRATEGIA B (Non Stratificata): Rimuove lo STESSO NUMERO ASSOLUTO di campioni
            calcolato in drop_macro, ma eliminando interamente intere microclassi (razze).
            """

            if not (0.0 <= percentage <= 1.0):
                raise ValueError("La percentuale deve essere compresa tra 0.0 e 1.0")

            rng = random.Random(self.seed)

            target_samples = [x for x in self.data_list if str(x[2]) == str(target_macro)]
            other_samples = [x for x in self.data_list if str(x[2]) != str(target_macro)]

            # Calcoliamo il "budget" esatto di campioni da rimuovere per un confronto equo
            total_target_count = len(target_samples)
            budget_to_remove = round(total_target_count * percentage)

            # Raggruppiamo i campioni per microclasse
            micro_groups = defaultdict(list)
            for sample in target_samples:
                micro_groups[sample[1]].append(sample)

            # Ordiniamo le chiavi per garantire che lo shuffle sia indipendente dal sistema operativo
            sorted_micro_keys = sorted(list(micro_groups.keys()))
            rng.shuffle(sorted_micro_keys)

            kept_micro_samples = []
            remaining_budget = budget_to_remove

            # Eliminiamo intere classi fino a esaurimento del budget
            for micro_key in sorted_micro_keys:
                samples = list(micro_groups[micro_key])

                if remaining_budget <= 0:
                    # Budget esaurito: questa classe viene tenuta interamente
                    kept_micro_samples.extend(samples)
                elif len(samples) <= remaining_budget:
                    # La classe è più piccola del budget rimasto: la eliminiamo TUTTA
                    remaining_budget -= len(samples)
                else:
                    # La classe è più grande del budget: eliminiamo solo la quota rimanente
                    rng.shuffle(samples)
                    num_to_keep = len(samples) - remaining_budget
                    kept_micro_samples.extend(samples[:num_to_keep])
                    remaining_budget = 0 # Budget azzerato

            return other_samples + kept_micro_samples
