import argparse
import os
import numpy as np
import pandas as pd
import torch

from models import MLP, TCN
from utils import (
    create_array_from_hdf,
    create_dataframe,
    create_labels_from_hdf,
    create_test_dataloader,
    drop_KAGRA,
    fix_seed
)


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate trained models on GW test data.")
    parser.add_argument("--hdf_file", nargs='*', required=True)
    parser.add_argument("--csv_file", default=None)
    parser.add_argument("--save_csv_path", default=None)
    parser.add_argument("--HEALPix", action='store_true', help='default: False')
    parser.add_argument("--use_KAGRA", action='store_true', help='default: False')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=int, default=1, help="1: MLP, 2: TCN, 3: Combined Ensemble")
    parser.add_argument("--weights_directory", default='weights')
    return parser.parse_args()


def predict_TCN(test_dataloader, n_sectors, input_size, weights_path, device, method):
    net = TCN(
        input_size=input_size, output_size=n_sectors,
        num_channels=[64] * 7, kernel_size=7, dropout=0.0
    )
    net.to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()

    y_preds = []
    corrects = 0

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)

            if method != 3:
                corrects += torch.sum(preds == targets.data)

            y_preds.append(outputs.cpu().numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    if method == 3:
        return y_pred
    else:
        accuracy = round((corrects.double() / len(test_dataloader.dataset)).item(), 5)
        return y_pred, accuracy


def predict_MLP(X_test, labels, n_sectors, use_KAGRA, weights_path, method):
    net = MLP(n_sectors=n_sectors, use_KAGRA=use_KAGRA)
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval()

    with torch.no_grad():
        y_pred = net(torch.from_numpy(X_test).float())
        _, y_pred_max = torch.max(y_pred, 1)
        y_pred_np = y_pred.numpy()
        y_pred_max = y_pred_max.numpy()

    if method == 3:
        return y_pred_np
    else:
        accuracy = float(np.sum(labels == y_pred_max) / len(X_test))
        return y_pred_np, accuracy


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_arguments()
    fix_seed(args.seed)

    input_size = 4 if args.use_KAGRA else 3
    detector = 'HLVK' if args.use_KAGRA else 'HLV'
    model_name = 'MLP' if args.method == 1 else 'TCN'

    df_ = None
    X = None

    if args.method in [1, 3]:
        if args.csv_file and os.path.exists(args.csv_file):
            df = pd.read_csv(args.csv_file)
        else:
            df = create_dataframe(args.hdf_file)

        if args.save_csv_path:
            os.makedirs(os.path.dirname(args.save_csv_path) or '.', exist_ok=True)
            df.to_csv(args.save_csv_path, index=None)

        if not args.use_KAGRA:
            df = drop_KAGRA(df)
        df_ = df.drop(['dec', 'ra', 'snr'], axis=1).values

    if args.method in [2, 3]:
        X = create_array_from_hdf(args.hdf_file, args.use_KAGRA).transpose(0, 2, 1)

    if not args.HEALPix:
        n_sectors_list = [2 * i * i for i in range(3, 11)]
    else:
        n_sectors_list = [12 * (4 ** i) for i in range(3)]

    accs = []
    print('Starting prediction...')

    for n_sectors in n_sectors_list:
        labels = create_labels_from_hdf(args.hdf_file, n_sectors, args.HEALPix)

        if args.method == 1:
            weights_path = f'{args.weights_directory}/MLP_{detector}{n_sectors}.pth'
            y_pred, accuracy = predict_MLP(
                df_, labels, n_sectors, args.use_KAGRA, weights_path, args.method
            )
        elif args.method == 2:
            weights_path = f'{args.weights_directory}/TCN_{detector}{n_sectors}.pth'
            test_dataloader = create_test_dataloader(X, labels, n_sectors, batch_size=200)
            y_pred, accuracy = predict_TCN(
                test_dataloader, n_sectors, input_size, weights_path, device, args.method
            )
        elif args.method == 3:
            mlp_weights = f'{args.weights_directory}/MLP_{detector}{n_sectors}.pth'
            y_pred1 = predict_MLP(df_, labels, n_sectors, args.use_KAGRA, mlp_weights, args.method)

            tcn_weights = f'{args.weights_directory}/TCN_{detector}{n_sectors}.pth'
            test_dataloader = create_test_dataloader(X, labels, n_sectors, batch_size=200)
            y_pred2 = predict_TCN(
                test_dataloader, n_sectors, input_size, tcn_weights, device, args.method
            )

            r = 0.6
            y_pred = r * y_pred1 + (1 - r) * y_pred2
            y_pred_max = np.argmax(y_pred, axis=1)
            accuracy = float(np.sum(labels == y_pred_max) / len(labels))
        else:
            raise ValueError("Select method 1 (MLP), 2 (TCN), or 3 (Combined)")

        accs.append(accuracy)
        print(f'n_sectors: {n_sectors} | Accuracy: {accuracy:.4f}')

    print('Overall Accuracies across resolutions:', accs)