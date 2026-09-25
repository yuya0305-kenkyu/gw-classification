import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"History CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 5))

    # Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=1.5)
    plt.plot(df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=1.5)
    plt.title('Learning Curve (Loss)', fontsize=13)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(df['train_acc'], label='Train Accuracy', color='#1f77b4', linewidth=1.5)
    plt.plot(df['val_acc'], label='Val Accuracy', color='#ff7f0e', linewidth=1.5)
    plt.title('Learning Curve (Accuracy)', fontsize=13)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training history from CSV log.")
    parser.add_argument("csv_path", help="Path to the training history CSV file")
    args = parser.parse_args()
    plot_learning_curve(args.csv_path)