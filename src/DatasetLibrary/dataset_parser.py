from config import ANNOTATIONS_PATH, IMAGES_PATH

def parse_annotation_file() -> list:
    parsed_data = []
    with open(ANNOTATIONS_PATH, mode='r') as list_file:
        for line in list_file:
            # Prende la riga, la pulisce e la splitta
            new_line = clean_and_split_line(line)
            if new_line is None:
                continue

            # 0 è il nome dell'immagine,
            # 1 la micro-categoria (id della classe )
            # 2 la macro-categoria (1 gatto, 2 cane)
            image_name = f"{new_line[0]}.jpg"
            micro_category = new_line[1]
            macro_category = new_line[2]

            tupla_micro_macro = (
                (IMAGES_PATH / image_name),
                micro_category,
                macro_category
            )

            parsed_data.append(tupla_micro_macro)

    return parsed_data





# Pulisce le stringhe. Returna la stringa splittata o none
def clean_and_split_line(line: str) -> list[str] | None:
    line = line.strip()

    if line.startswith('#') or not line:
        return None

    # Splitta la stringa e la returna
    line_splitted = line.split()
    return line_splitted
