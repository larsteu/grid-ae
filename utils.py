import pandas as pd
import json
import numpy as np
import torch

import config
import matplotlib
import matplotlib.pyplot as plt


# matplotlib.use("TkAgg")


def load_dataset(path):
    return pd.read_csv(path, quotechar='"', quoting=1)


def preprocess_dataset(dataset, cat_cols):
    dataset = config.specific_transformations(dataset)
    processed_dataset, mappings = categories_to_numerical(dataset, cat_cols)
    with open(config.DATASET_MAPPINGS_FILE, "w") as outfile:
        json.dump(mappings, outfile)

    return processed_dataset


def categories_to_numerical(dataset, cat_cols):
    mappings = {}
    for col in cat_cols:
        vals = dataset[col].unique()
        counter = 0
        mapping = {}
        for val in vals:
            mapping[val] = counter
            dataset = dataset.replace(val, counter)
            counter = counter + 1
        mappings[col] = mapping

    return dataset, mappings


def numerical_to_categories(dataset, mapping_file):
    with open(mapping_file) as json_file:
        mappings = json.load(json_file)
    for column_name in mappings:
        for categorical_value in mappings[column_name]:
            dataset[column_name] = dataset[column_name].replace(
                mappings[column_name][categorical_value], categorical_value
            )

    return dataset


def denormalize_dataset(dataset, normalization_info_file, scaler=None):
    if scaler:
        target = dataset[config.TARGET_COL].values
        p = dataset.iloc[:, : config.COLUMNS_MAP[config.TARGET_COL]]
        dataset = pd.DataFrame(
            scaler.inverse_transform(p), columns=config.COLUMNS_LST_WITHOUT_TARGET
        )
        dataset[config.TARGET_COL] = target
        return dataset

    with open(normalization_info_file) as json_file:
        normalization_info = json.load(json_file)

    for column_name in normalization_info:
        max_val = normalization_info[column_name]["max"]
        min_val = normalization_info[column_name]["min"]

        dataset[column_name] = dataset[column_name] * (max_val - min_val) + min_val
    return dataset


def normalize_dataset(dataset, normalization_info_file_path, scaler=None):
    if scaler:
        target = dataset[config.TARGET_COL].values
        p = dataset.drop(config.TARGET_COL, axis=1)
        dataset = pd.DataFrame(
            scaler.fit_transform(p), columns=config.COLUMNS_LST_WITHOUT_TARGET
        )
        dataset[config.TARGET_COL] = target
        return dataset

    normalization_info = {}
    for col in dataset.columns:
        max_val = int(dataset[col].max())
        min_val = int(dataset[col].min())

        values = {"max": max_val, "min": min_val}
        normalization_info[col] = values

        dataset[col] = (dataset[col] - min_val) / (max_val - min_val)

    with open(normalization_info_file_path, "w") as outfile:
        json.dump(normalization_info, outfile)

    return dataset


def plot_latent_space(dataloader, model):
    points = []
    labels = []
    for i, data in enumerate(dataloader):
        inputs = data
        inputs = inputs.float()

        labels.extend(
            set_label(inputs, inputs[:, config.COLUMNS_MAP[config.TARGET_COL]])
        )

        with torch.no_grad():
            output = model.encode(inputs).numpy()
        points.extend(output)

    points = np.array(points)
    if points.shape[1] < 3:
        plt.scatter(x=points[:, 0], y=points[:, 1], c=labels, s=1, cmap="jet")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=1, cmap="jet")
    plt.show()


def plot_points(points):
    plt.scatter(x=points[:, 0], y=points[:, 1], s=1)
    plt.show()


def plot_points_two_graphs(points1, points2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(points1[:, 0], points1[:, 1], s=1)
    ax1.set_title("Encoded dataset (a)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y", rotation=0)
    ax2.scatter(points2[:, 0], points2[:, 1], s=1)
    ax2.set_title("Sampled points (b)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y", rotation=0)
    plt.show()


def set_label(batch, label):
    labels = []
    if config.LABEL_TYPE == config.TARGET_COL:
        return label.numpy()
    for val in batch:
        labels.append(val[config.COLUMNS_MAP[config.LABEL_TYPE]])
    return labels


def generate_example_output(dataloader, autoencoder):
    inputs = next(iter(dataloader))
    inputs = inputs.float()
    print(inputs[0])
    outputs = autoencoder.process(inputs)
    print(outputs[0])


def create_new_samples(
    encoded_values,
    min_x=0,
    max_x=config.GRID_DENSITY_FACTOR,
    min_y=0,
    max_y=config.GRID_DENSITY_FACTOR,
):
    res = []
    lst_encoded = encoded_values.tolist()
    for i in range(len(lst_encoded)):
        lst_encoded[i][0] = round(lst_encoded[i][0], 4)
        lst_encoded[i][1] = round(lst_encoded[i][1], 4)

    if config.AUTOENCODER_ENCODING_DIMENSION == 2:
        for i in range(min_x, max_x, 1):
            x = i / config.GRID_DENSITY_FACTOR
            for j in range(min_y, max_y, 1):
                y = j / config.GRID_DENSITY_FACTOR
                x_ = round(x, 4)
                y_ = round(y, 4)
                if not ([x_, y_] in lst_encoded):
                    res.append(np.array([x, y]))
    elif config.AUTOENCODER_ENCODING_DIMENSION == 4:
        for l in range(min_x, max_x, 1):
            q = l / config.GRID_DENSITY_FACTOR
            for k in range(min_x, max_x, 1):
                z = k / config.GRID_DENSITY_FACTOR
                for i in range(min_x, max_x, 1):
                    x = i / config.GRID_DENSITY_FACTOR
                    for j in range(min_y, max_y, 1):
                        y = j / config.GRID_DENSITY_FACTOR
                        q_ = round(q, 4)
                        z_ = round(z, 4)
                        x_ = round(x, 4)
                        y_ = round(y, 4)
                        if not ([q_, z_, x_, y_] in lst_encoded):
                            res.append(np.array([q_, z_, x_, y_]))

    return np.array(res)


def save_sample_dataset(sample_points, scaler):
    dataset = pd.DataFrame(sample_points, columns=config.COLUMNS_LST)
    dataset = denormalize_dataset(dataset, "data/normalization_info.json", scaler)
    dataset[config.TARGET_COL] = dataset[config.TARGET_COL]
    dataset = dataset.drop_duplicates()
    dataset.to_csv("data/datasets/new_dataset.csv", index=False)
