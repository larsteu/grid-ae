## Smart Factory Dataset config ##
AUTOENCODER_INPUT = 40
DOWNSTREAM_INPUT = 39

DATASET_TRAIN_PATH = (
    "data/datasets/soccer/soccer_train.csv"  # Path of the used dataset for training(as csv-file)
)
DATASET_TEST_PATH = (
    "data/datasets/soccer/soccer_test.csv"  # Path of the used dataset for testing(as csv-file)
)
COLUMNS_MAP = {
    "id_x": 0,
    "height": 1,
    "weight": 2,
    "overall_rating": 3,
    "potential": 4,
    "preferred_foot": 5,
    "attacking_work_rate": 6,
    "crossing": 7,
    "finishing": 8,
    "heading_accuracy": 9,
    "short_passing": 10,
    "volleys": 11,
    "dribbling": 12,
    "curve": 13,
    "free_kick_accuracy": 14,
    "long_passing": 15,
    "ball_control": 16,
    "acceleration": 17,
    "sprint_speed": 18,
    "agility": 19,
    "reactions": 20,
    "balance": 21,
    "shot_power": 22,
    "jumping": 23,
    "stamina": 24,
    "strength": 25,
    "long_shots": 26,
    "aggression": 27,
    "interceptions": 28,
    "positioning": 29,
    "vision": 30,
    "penalties": 31,
    "marking": 32,
    "standing_tackle": 33,
    "sliding_tackle": 34,
    "gk_diving": 35,
    "gk_handling": 36,
    "gk_kicking": 37,
    "gk_positioning": 38,
    "gk_reflexes": 39
}  # List of all columns with its corresponding index
COLUMNS_LST = [
    "id_x",
    "height",
    "weight",
    "overall_rating",
    "potential",
    "preferred_foot",
    "attacking_work_rate",
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "free_kick_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes"
]
COLUMNS_LST_WITHOUT_TARGET = [
    "id_x",
    "height",
    "weight",
    "potential",
    "preferred_foot",
    "attacking_work_rate",
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "free_kick_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes"
]
CATEGORICAL_COLUMNS = [
    "preferred_foot",
    "attacking_work_rate",
]  # List of columns that contain categorical values (need to be converted to numerical values)
TARGET_COL = "overall_rating"  # column name of the target column (for the down-stream task)
LABEL_TYPE = "overall_rating"  # column that is used for color mapping the latent space plot

# removed: "player_name": 1,
#           "birthday": 2,
#           "date": 5

DATASET_MAPPINGS_FILE = "data/dataset_mappings.json"  # Filename of the file containing mappings between categorical and numerical values


def specific_transformations(dataset):
    dataset = dataset.drop(["player_name", "birthday", "date"], axis=1)
    dataset = dataset.dropna()
    return dataset
