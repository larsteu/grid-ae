import numpy as np
from torch.utils.data import DataLoader
import config
import torch
import os

from sklearn.preprocessing import StandardScaler
from dataset import AutoencoderDataset, ClassifierDataset
from losses import RMSELoss, CustomGridLoss
from models.autoencoder import Autoencoder
from models.downstream_model import Classifier
from models.encoder_decoder_1 import Encoder, Decoder
from utils import (
    load_dataset,
    preprocess_dataset,
    plot_latent_space,
    generate_example_output,
    create_new_samples,
    plot_points,
    save_sample_dataset,
    plot_points_two_graphs
)


def create_sample_dataset(origin_dataloader, autoencoder, scaler):
    # Encode whole dataset
    encoded_vals = []
    original_data = []
    for i, data in enumerate(origin_dataloader):
        input_val = data
        input_val = input_val.float()

        encoded_val = autoencoder.encode(input_val).numpy()

        original_data.extend(input_val.numpy())
        encoded_vals.extend(encoded_val)

    encoded_vals = np.array(encoded_vals)

    # Create new samples base upon the encoded dataset
    min_x = int(float(input("Min x: ")) * config.GRID_DENSITY_FACTOR)
    max_x = int(float(input("Max x: ")) * config.GRID_DENSITY_FACTOR)
    min_y = int(float(input("Min y: ")) * config.GRID_DENSITY_FACTOR)
    max_y = int(float(input("Max y: ")) * config.GRID_DENSITY_FACTOR)

    sampled_values = create_new_samples(encoded_vals, min_x, max_x, min_y, max_y)

    # Plot the encoded dataset and the new sampled points
    plot_points(sampled_values)
    plot_points(np.concatenate((encoded_vals, sampled_values)))

    # Decode the samples and get their predicted label
    sampled_data = decode_data(sampled_values, autoencoder)
    #save_sample_dataset(sampled_data, scaler)

    plot_points_two_graphs(encoded_vals, sampled_values)

    # Create a new dataset with the decoded original data and the decoded samples
    down_stream_dataset = ClassifierDataset(
        np.concatenate((np.array(original_data), sampled_data)),
        scaler,
        config.COLUMNS_LST,
        config.TARGET_COL,
        config.COLUMNS_MAP[config.TARGET_COL],
    )
    return down_stream_dataset


def decode_data(data, model):
    decoded_vals = []
    for datapoint in data:
        input_val = torch.tensor(np.array([datapoint])).float()
        decoded_val = model.decode(input_val).numpy()
        decoded_vals.extend(decoded_val)

    return np.array(decoded_vals)


def main():
    # Initialize Scaler
    scaler = StandardScaler()

    # Load and preprocess dataset
    dataset_train = load_dataset(config.DATASET_TRAIN_PATH)
    dataset_train = preprocess_dataset(dataset_train, config.CATEGORICAL_COLUMNS)

    dataset_test = load_dataset(config.DATASET_TEST_PATH)
    dataset_test = preprocess_dataset(dataset_test, config.CATEGORICAL_COLUMNS)

    # Create pytorch dataset and dataloader
    autoencoder_dataset_train = AutoencoderDataset(
        dataset_train, config.CATEGORICAL_COLUMNS, scaler, True
    )

    autoencoder_dataset_test = AutoencoderDataset(
        dataset_test, config.CATEGORICAL_COLUMNS, scaler, True
    )

    if config.TEST_SPARSE_DATA:
        autoencoder_dataset_train, _ = torch.utils.data.random_split(
            autoencoder_dataset_train,
            (config.SPARSE_DATA_FACTOR, 1-config.SPARSE_DATA_FACTOR),
            generator=torch.Generator().manual_seed(123)
        )

    autoencoder_dataloader = DataLoader(
        autoencoder_dataset_train, batch_size=config.BATCH_SIZE, shuffle=True
    )

    # Create autoencoder model, optimizer and loss function
    autoencoder = Autoencoder(
        Encoder(
            config.AUTOENCODER_INPUT_DIMENSION, config.AUTOENCODER_ENCODING_DIMENSION
        ),
        Decoder(
            config.AUTOENCODER_INPUT_DIMENSION, config.AUTOENCODER_ENCODING_DIMENSION
        ),
        config.GRID_DENSITY_FACTOR,
    )

    autoencoder_optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=config.AUTOENCODER_LR
    )
    autoencoder_loss_fn = CustomGridLoss()

    if os.path.exists(config.AUTOENCODER_PATH + "/" + config.AUTOENCODER_NAME):
        autoencoder.load_model(
            autoencoder_optimizer,
            config.AUTOENCODER_LR,
            config.AUTOENCODER_PATH,
            config.AUTOENCODER_NAME,
        )

    # Train autoencoder
    for epoch in range(config.AUTOENCODER_EPOCHS):
        autoencoder.train_epoch(
            epoch,
            autoencoder_dataloader,
            autoencoder_optimizer,
            autoencoder_loss_fn,
        )
    autoencoder.save_model(
        autoencoder_optimizer, config.AUTOENCODER_PATH, config.AUTOENCODER_NAME
    )

    # Plot latent space and create an example output
    plot_latent_space(autoencoder_dataloader, autoencoder)
    generate_example_output(autoencoder_dataloader, autoencoder)

    # Encode dataset, create new samples based on the dataset and create a new large dataset
    down_stream_dataset = create_sample_dataset(
        autoencoder_dataloader, autoencoder, scaler
    )
    print(len(autoencoder_dataset_train))
    print(len(down_stream_dataset))

    down_stream_dataloader = DataLoader(
        down_stream_dataset,
        batch_size=config.DOWN_STREAM_MODEL_BATCH_SIZE,
        shuffle=True,
    )

    # Create down-stream model, optimizer and loss function
    down_stream_model = Classifier(config.DOWN_STREAM_MODEL_INPUT_DIMENSION)
    loss_fn_down_stream = RMSELoss()

    down_stream_optimizer = torch.optim.Adam(
        down_stream_model.parameters(), lr=config.DOWN_STREAM_MODEL_LR
    )

    down_stream_dataloader_test = DataLoader(
        autoencoder_dataset_test,
        batch_size=config.DOWN_STREAM_MODEL_BATCH_SIZE,
        shuffle=True,
    )

    # Train down-stream model
    for epoch in range(config.DOWN_STREAM_MODEL_EPOCHS):
        down_stream_model.train_epoch(
            epoch, down_stream_dataloader, down_stream_optimizer, loss_fn_down_stream
        )
        down_stream_model.eval_model(
            down_stream_dataloader_test,
            loss_fn_down_stream,
            config.COLUMNS_MAP[config.TARGET_COL],
        )

    # Eval down-stream model
    for epoch in range(5):
        down_stream_model.eval_model(
            down_stream_dataloader_test,
            loss_fn_down_stream,
            config.COLUMNS_MAP[config.TARGET_COL],
        )

    # To beat: 0.325, current results: 0.387
    # Adults: Results: 0-1; 0-1 -> 0.353 (2 epochs of training)
    # Smart_factory: Baseline:0.161; Model: 0.146
    # Soccer: Baseline: 0.578; VAE: 0.57; Model: 0.553

    # down_stream_model.save_model(
    #    down_stream_optimizer,
    #    config.DOWN_STREAM_MODEL_PATH,
    #    config.DOWN_STREAM_MODEL_NAME,
    # )


if __name__ == "__main__":
    main()
