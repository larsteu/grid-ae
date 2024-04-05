## Adults Dataset config ##
AUTOENCODER_INPUT = 15
DOWNSTREAM_INPUT = 14

DATASET_TRAIN_PATH = (
    "data/datasets/adults/train.csv"  # Path of the used dataset for training(as csv-file)
)
DATASET_TEST_PATH = (
    "data/datasets/adults/test.csv"  # Path of the used dataset for testing(as csv-file)
)

COLUMNS_MAP = {
    "age": 0,
    "workclass": 1,
    "fnlwgt": 2,
    "education": 3,
    "educational_num": 4,
    "marital_status": 5,
    "occupation": 6,
    "relationship": 7,
    "race": 8,
    "gender": 9,
    "capital_gain": 10,
    "capital_loss": 11,
    "hours_per_week": 12,
    "native_country": 13,
    "income": 14,
}  # List of all columns with its corresponding index
COLUMNS_LST = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "educational_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
COLUMNS_LST_WITHOUT_TARGET = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "educational_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]
CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native_country",
]  # List of columns that contain categorical values (need to be converted to numerical values)
TARGET_COL = "income"  # column name of the target column (for the down-stream task)
LABEL_TYPE = "income"  # column that is used for color mapping the latent space plot

DATASET_MAPPINGS_FILE = "data/dataset_mappings.json"  # Filename of the file containing mappings between categorical and numerical values


def specific_transformations(dataset):
    return dataset
