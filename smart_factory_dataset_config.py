## Smart Factory Dataset config ##
AUTOENCODER_INPUT = 19
DOWNSTREAM_INPUT = 18

DATASET_TRAIN_PATH = (
    "data/datasets/smart_factory/smart_factory_train.csv"  # Path of the used dataset (as csv-file)
)
DATASET_TEST_PATH = (
    "data/datasets/smart_factory/smart_factory_test.csv"  # Path of the used dataset for testing(as csv-file)
)
COLUMNS_MAP = {
    "labels" : 0,
    "i_w_blo_weg": 1,
    "o_w_blo_power": 2,
    "o_w_blo_voltage": 3,
    "i_w_bhl_weg": 4,
    "o_w_bhl_power": 5,
    "o_w_bhl_voltage": 6,
    "i_w_bhr_weg": 7,
    "o_w_bhr_power": 8,
    "o_w_bhr_voltage": 9,
    "i_w_bru_weg": 10,
    "o_w_bru_power": 11,
    "o_w_bru_voltage": 12,
    "i_w_hr_weg": 13,
    "o_w_hr_power": 14,
    "o_w_hr_voltage": 15,
    "i_w_hl_weg": 16,
    "o_w_hl_power": 17,
    "o_w_hl_voltage": 18
}  # List of all columns with its corresponding index
COLUMNS_LST = [
    "labels",
    "i_w_blo_weg",
    "o_w_blo_power",
    "o_w_blo_voltage",
    "i_w_bhl_weg",
    "o_w_bhl_power",
    "o_w_bhl_voltage",
    "i_w_bhr_weg",
    "o_w_bhr_power",
    "o_w_bhr_voltage",
    "i_w_bru_weg",
    "o_w_bru_power",
    "o_w_bru_voltage",
    "i_w_hr_weg",
    "o_w_hr_power",
    "o_w_hr_voltage",
    "i_w_hl_weg",
    "o_w_hl_power",
    "o_w_hl_voltage"
]
COLUMNS_LST_WITHOUT_TARGET = [
    "i_w_blo_weg",
    "o_w_blo_power",
    "o_w_blo_voltage",
    "i_w_bhl_weg",
    "o_w_bhl_power",
    "o_w_bhl_voltage",
    "i_w_bhr_weg",
    "o_w_bhr_power",
    "o_w_bhr_voltage",
    "i_w_bru_weg",
    "o_w_bru_power",
    "o_w_bru_voltage",
    "i_w_hr_weg",
    "o_w_hr_power",
    "o_w_hr_voltage",
    "i_w_hl_weg",
    "o_w_hl_power",
    "o_w_hl_voltage"
]
CATEGORICAL_COLUMNS = []  # List of columns that contain categorical values (need to be converted to numerical values)
TARGET_COL = "labels"  # column name of the target column (for the down-stream task)
LABEL_TYPE = "labels"  # column that is used for color mapping the latent space plot

DATASET_MAPPINGS_FILE = "data/dataset_mappings.json"  # Filename of the file containing mappings between categorical and numerical values


def specific_transformations(dataset):
    return dataset
