import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP, TCN
from utils import (
    create_array_from_hdf,
    create_augmented_dataloader,
    create_dataframe,
    create_labels_from_hdf,
    drop_KAGRA,
    fix_seed
)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train MLP or TCN model on gravitational wave data.")
    parser.add_argument("--hdf_file", nargs='*', required=True, help="Input HDF5 file paths")
    parser.add_argument("--csv_file", default=None, help="Pre-extracted CSV file")
    parser.add_argument("--save_csv_path", default=None, help="Path to save extracted features CSV")
    parser.add_argument("--HEALPix", action='store_true', help="Use HEALPix sky tessellation (default: False)")
    parser.add_argument("--method", type=int, default=1, help="Architecture selection: 1=MLP, 2=TCN")
    parser.add_argument("--use_KAGRA", action='store_true', help="Use 4 detectors HLVK (default: HLV 3 detectors)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--training_size", type=int, default=200000)
    parser.add_argument("--weights_directory", default='weights')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def train_model(net, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    max_acc = 0.0

    patience = 20
    no_improve_count = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}', end='\t')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            dataset_len = len(dataloaders_dict[phase].dataset)
            epoch_loss = epoch_loss / dataset_len
            epoch_acc = (epoch_corrects.double() / dataset_len).item()

            if phase == 'train':
                train_loss.append(round(epoch_loss, 5))
                train_acc.append(round(epoch_acc, 5))
                print(f'train loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}', end='\t')
            else:
                val_loss.append(round(epoch_loss, 5))
                val_acc.append(round(epoch_acc, 5))
                print(f'val loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}')

                if epoch_acc > max_acc:
                    max_acc = epoch_acc
                    torch.save(net.state_dict(), weights_path)
                    print('Saved the weights.')
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        if no_improve_count >= patience:
            print(f'\nEarly Stopping: {patience} epochs without validation accuracy improvement.')
            print(f'Best Accuracy: {max_acc:.4f}')
            break

        scheduler.step(epoch_loss)

    return {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    args = get_arguments()
    fix_seed(args.seed)

    input_size = 4 if args.use_KAGRA else 3
    detector = 'HLVK' if args.use_KAGRA else 'HLV'
    model_name = 'MLP' if args.method == 1 else 'TCN'

    print('Start preprocessing...')
    if args.method == 1:
        X = np.array([])
        if args.csv_file and os.path.exists(args.csv_file):
            df = pd.read_csv(args.csv_file)
        else:
            df = create_dataframe(args.hdf_file)

        if args.save_csv_path:
            os.makedirs(os.path.dirname(args.save_csv_path) or '.', exist_ok=True)
            df.to_csv(args.save_csv_path, index=None)

        if not args.use_KAGRA:
            df = drop_KAGRA(df)
    elif args.method == 2:
        X = create_array_from_hdf(args.hdf_file, args.use_KAGRA).transpose(0, 2, 1)
        df = pd.DataFrame([])
    else:
        raise ValueError("Invalid method: choose 1 (MLP) or 2 (TCN)")

    if not args.HEALPix:
        n_sectors_list = [2 * i * i for i in range(3, 11)]
    else:
        n_sectors_list = [12 * (4 ** i) for i in range(3)]

    print('Start training loop across sector resolutions...')
    for n_sectors in n_sectors_list:
        print('#' * 40)
        print(f'Target Number of Sectors: {n_sectors}')
        print('#' * 40)

        os.makedirs(args.weights_directory, exist_ok=True)
        weights_path = os.path.join(args.weights_directory, f'{model_name}_{detector}{n_sectors}.pth')
        labels = create_labels_from_hdf(args.hdf_file, n_sectors, args.HEALPix)

        if args.method == 1:
            net = MLP(n_sectors=n_sectors, use_KAGRA=args.use_KAGRA)
        else:
            net = TCN(
                input_size=input_size,
                output_size=n_sectors,
                num_channels=[64] * 7,
                kernel_size=7,
                dropout=args.dropout
            )

        dataloaders_dict = create_augmented_dataloader(
            df=df, arr=X, labels=labels,
            batch_size=args.batch_size, training_size=args.training_size
        )

        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        history = train_model(
            net=net,
            dataloaders_dict=dataloaders_dict,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.epoch,
            weights_path=weights_path
        )

        history_df = pd.DataFrame(history)
        history_csv_path = os.path.join(args.weights_directory, f'history_{model_name}_{detector}{n_sectors}.csv')
        history_df.to_csv(history_csv_path, index=False)
        print(f'Saved training history to {history_csv_path}')