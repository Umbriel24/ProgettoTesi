"""
Script di verifica: mostra che drop_micro elimina intere razze,
mentre drop_macro riduce proporzionalmente tutte le razze.

Uso: aggiusta gli import in base a dove sono i tuoi moduli
(DatasetLibrary.dataset_parser, DatasetLibrary.dataset_splitter, ecc.)
"""

from collections import Counter

from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data
from DatasetLibrary.dataset_dropper import DatasetDropper
import config


def conta_per_razza(data_list):
    # Indice 1 = micro_category (id razza)
    return Counter(sample[1] for sample in data_list)


def stampa_confronto(nome_test, prima, dopo):
    print(f"\n===== {nome_test} =====")
    print(f"{'razza':<10}{'prima':>10}{'dopo':>10}{'diff %':>10}")

    razze_sparite = []
    razze_intatte = []

    for razza in sorted(prima.keys(), key=lambda x: int(x)):
        n_prima = prima[razza]
        n_dopo = dopo.get(razza, 0)
        diff_pct = (1 - n_dopo / n_prima) * 100 if n_prima else 0
        print(f"{razza:<10}{n_prima:>10}{n_dopo:>10}{diff_pct:>9.1f}%")

        if n_dopo == 0:
            razze_sparite.append(razza)
        elif n_dopo == n_prima:
            razze_intatte.append(razza)

    print(f"\nRazze COMPLETAMENTE ELIMINATE (dopo=0): {razze_sparite}")
    print(f"Razze RIMASTE INTATTE (dopo=prima):      {razze_intatte}")


if __name__ == "__main__":
    parsed_data = parse_annotation_file()
    train_subset, val_subset, test_subset = split_parsed_data(
        parsed_data=parsed_data,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED
    )

    target_macro_class = '2'
    percentage = 0.20  # 20%, scegli tu

    conteggio_prima = conta_per_razza(
        [s for s in train_subset if str(s[2]) == target_macro_class]
    )

    # --- Test drop_macro ---
    dropper_macro = DatasetDropper(train_subset, seed=config.SEED)
    dopo_macro = dropper_macro.drop_macro(target_macro=target_macro_class, percentage=percentage)
    conteggio_dopo_macro = conta_per_razza(
        [s for s in dopo_macro if str(s[2]) == target_macro_class]
    )
    stampa_confronto("DROP_MACRO (atteso: nessuna razza a zero)", conteggio_prima, conteggio_dopo_macro)

    # --- Test drop_micro ---
    dropper_micro = DatasetDropper(train_subset, seed=config.SEED)
    dopo_micro = dropper_micro.drop_micro(target_macro=target_macro_class, percentage=percentage)
    conteggio_dopo_micro = conta_per_razza(
        [s for s in dopo_micro if str(s[2]) == target_macro_class]
    )
    stampa_confronto("DROP_MICRO (atteso: alcune razze a zero, altre intatte)", conteggio_prima, conteggio_dopo_micro)

    # --- Test idempotenza (stesso seed => stesso risultato) ---
    dropper_micro_2 = DatasetDropper(train_subset, seed=config.SEED)
    dopo_micro_2 = dropper_micro_2.drop_micro(target_macro=target_macro_class, percentage=percentage)
    identico = sorted(dopo_micro, key=str) == sorted(dopo_micro_2, key=str)
    print(f"\nIdempotenza drop_micro (stesso seed -> stesso risultato): {identico}")