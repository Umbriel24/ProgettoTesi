from sklearn.model_selection import train_test_split

def split_parsed_data(parsed_data, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15, seed = 777):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) >= 0.01:
        raise Exception("La somma dei valori di train, val e test non è 1.0.")

#Isoliamo prima il 70% e 30%, successivamente isoliamo il 30% in due 15%

    # Estrazione label di stratificazione micro-categoria, ovvero le razze. Indice 1
    # List comprehension prestazione O(N)
    labels_micro = [tupla[1] for tupla in parsed_data]

    # Split 1: Isoliamo il 70% e 30%
    test_val_ratio = val_ratio + test_ratio # 30%
    train_data, temp_data, _, temp_label = train_test_split(
        parsed_data,
        labels_micro,
        test_size=test_val_ratio,
        stratify=labels_micro,
        random_state=seed
    )

    # Split 2: Isoliamo il 30% in due 15%
    val_data, test_data = train_test_split(
        temp_data,
        test_size = 0.5,
        stratify=temp_label,
        random_state=seed
    )

    return train_data, val_data, test_data
