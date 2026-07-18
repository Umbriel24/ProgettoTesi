import zipfile

import config


def zippa_file_kaggle():
    archive_path = config.PERSISTANCE_PATH
    archive = zipfile.ZipFile("KaggleModels.zip", 'a')


    for file in archive_path.iterdir():
        archive.write(file, compress_type=zipfile.ZIP_DEFLATED)

    archive.close()


zippa_file_kaggle()