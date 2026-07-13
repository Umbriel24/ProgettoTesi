import unittest

from DatasetLibrary.dataset_parser import parse_annotation_file
from DatasetLibrary.dataset_splitter import split_parsed_data

class TestDataset(unittest.TestCase):
    def test_parser(self):

        # 1. Arrange: prepara
        risultato_atteso = 7349

        # 2. ACT
        parsed_data = parse_annotation_file()

        # 3. ASSERT
        self.assertEqual(len(parsed_data), risultato_atteso)


    def test_splitter(self):

        # 1. Arrange
        campioni_train = 5144
        campioni_validation = 1102
        campioni_test = 1102

        # 2. Act
        parsed_data = parse_annotation_file()
        train, val, test = split_parsed_data(parsed_data=parsed_data)

        # 3. Assert
        differenza_train = abs(campioni_train - len(train))
        differenza_validation = abs(campioni_validation - len(val))
        differenza_test = abs(campioni_test - len(test))

        self.assertLessEqual(differenza_train, 10)
        self.assertLessEqual(differenza_validation, 10)
        self.assertLessEqual(differenza_test, 10)
