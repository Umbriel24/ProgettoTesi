import zipfile

import config


def zippa_file_kaggle():
    archive_path = config.PERSISTANCE_PATH
    archive = zipfile.ZipFile("KaggleModels.zip", 'a')


    for file in archive_path.iterdir():
        archive.write(file, compress_type=zipfile.ZIP_DEFLATED)

    archive.close()



# Prende tutit i file nella cartella persistenza. Se iniziano con report,
# li inseriscen nello zip.
def zippa_report_file_kaggle():
    archive_path = config.PERSISTANCE_PATH
    archive = zipfile.ZipFile("KaggleReports.zip", 'a')
    for file in archive_path.iterdir():
        if file.name.startswith("report"):
            archive.write(file, compress_type=zipfile.ZIP_DEFLATED)

    archive.close()


def zippa_all_csv_file_kaggle():
    archive_path = config.PERSISTANCE_PATH
    archive = zipfile.ZipFile("KaggleCSV.zip", 'a')

    for file in archive_path.iterdir():
        if file.name.endswith(".csv"):
            archive.write(file, compress_type=zipfile.ZIP_DEFLATED)


zippa_file_kaggle()