from config import ANNOTATIONS_PATH, IMAGES_PATH

def parse_annotation_file() -> list:
    parsed_data = []
    with open(ANNOTATIONS_PATH, mode='r') as list_file:
        for line in list_file:
            # Prende la riga, la pulisce e la splitta
            newLine = clean_and_split_line(line)
            if(newLine is None):
                continue

            # 0 è il nome dell'immagine,
            # 1 la micro-categoria (id della classe )
            # 2 la macro-categoria (1 gatto, 2 cane)
            image_name = f"{newLine[0]}.jpg"
            micro_category = newLine[1]
            macro_category = newLine[2]

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
