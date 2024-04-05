import soccer_dataset_config as database_config

## Config ##

## Tests ##
TEST_SPARSE_DATA = True
SPARSE_DATA_FACTOR = 0.01

## Autoencoder ##
AUTOENCODER_INPUT_DIMENSION = (
    database_config.AUTOENCODER_INPUT  # Dimension of the autoencoder input (number of columns of the table)
)
AUTOENCODER_ENCODING_DIMENSION = 2  # Dimension of the autoencoder encoding dimension

AUTOENCODER_PATH = "data/models"  # Path of the autoencoder save file
AUTOENCODER_NAME = "autoencoder"  # Name of the autoencoder save file

BATCH_SIZE = 64  # Batch size used for the autoencoder

AUTOENCODER_LR = 0.001  # Learning rate of the autoencoder

AUTOENCODER_EPOCHS = 800
# epochs for the autoencoder

GRID_DENSITY_FACTOR = 400  # Defines the density of the grid (distance between two closest point: 1/GRID_DENSITY_FACTOR)

## DOWN_STREAM_MODEL ##
DOWN_STREAM_MODEL_INPUT_DIMENSION = database_config.DOWNSTREAM_INPUT  # Dimension of the down-stream model input (must be identical to autoencoder input dimension)

DOWN_STREAM_MODEL_PATH = "data/models"  # Path of the down-stream model save file
DOWN_STREAM_MODEL_NAME = "down_stream_model"  # Name of the down-stream model save file

DOWN_STREAM_MODEL_LR = 0.00001  # Learning rate of the down-stream model

DOWN_STREAM_MODEL_BATCH_SIZE = 64  # # Batch size used for the down stream model

DOWN_STREAM_MODEL_EPOCHS = 20  # epochs for the down-stream model

## Dataset - DO NOT CHANGE HERE - USE DATASET CONFIG FILE##
DATASET_TRAIN_PATH = database_config.DATASET_TRAIN_PATH  # Path of the used dataset (as csv-file)
DATASET_TEST_PATH = database_config.DATASET_TEST_PATH  # Path of the used dataset (as csv-file)
COLUMNS_MAP = (
    database_config.COLUMNS_MAP
)  # List of all columns with its corresponding index
COLUMNS_LST = database_config.COLUMNS_LST
COLUMNS_LST_WITHOUT_TARGET = database_config.COLUMNS_LST_WITHOUT_TARGET
CATEGORICAL_COLUMNS = (
    database_config.CATEGORICAL_COLUMNS
)  # List of columns that contain categorical values (need to be converted to numerical values)
TARGET_COL = (
    database_config.TARGET_COL
)  # column name of the target column (for the down-stream task)
LABEL_TYPE = (
    database_config.LABEL_TYPE
)  # column that is used for color mapping the latent space plot

DATASET_MAPPINGS_FILE = (
    database_config.DATASET_MAPPINGS_FILE
)  # Filename of the file containing mappings between categorical and numerical values


def specific_transformations(dataset):
    return database_config.specific_transformations(dataset)
