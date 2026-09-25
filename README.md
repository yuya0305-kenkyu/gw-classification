# GW-Classification: Gravitational Wave Signal Classifier

PyTorchを用いた重力波（Gravitational Wave）信号分類のための機械学習パイプラインです。
MLP（多層パーセプトロン）および TCN（Temporal Convolutional Network）を用いて、LIGO（Hanford, Livingston）、Virgo、および KAGRA（HLVK）の各検出器から得られるひずみデータ（Strain data）の特徴を学習し、信号の到来方向（セクター）を分類・予測します。

## 特徴 (Features)

*   **複数モデルのサポート**: 軽量な MLP と、時系列データに強い TCN（Global Max Pooling 実装済み）の比較・アンサンブル推論が可能。
*   **KAGRA対応**: `--use_KAGRA` オプションにより、標準の HLV（3検出器）から HLVK（4検出器）への拡張解析が可能。
*   **O4観測ラン準拠のシミュレーション**: データ生成スクリプトではO4（第4期観測実行）の感度曲線モデルを想定。実観測データに近いノイズ環境下で学習・検証が可能。
*   **堅牢なデータ拡張**: TCN学習時におけるタイムシフト、振幅スケーリング、ガウシアンノイズ注入（Data Augmentation）による汎化性能の向上。

## ディレクトリ構成 (Directory Structure)

```text
gw-classification/
├── data_generation/
│   ├── generate_data.py     # 学習・検証・テスト用HDF5データ生成スクリプト
│   └── sensitivities/       # O4/KAGRA等のPSD（ノイズ感度）ファイル群
├── models/
│   ├── mlp.py               # MLPモデル定義
│   └── tcn.py               # TCNモデル定義
├── utils/
│   ├── dataset.py           # Dataset, DataLoaderの定義
│   └── preprocess.py        # 特徴量抽出、データ前処理
├── train.py                 # モデル学習用スクリプト
├── predict.py               # 推論用スクリプト
├── plot_history.py          # 学習曲線の可視化スクリプト
├── requirements.txt         # 依存ライブラリ一覧
└── notebooks/
    └── Tutorial_Colab.ipynb # Google Colab用チュートリアルノートブック

```
## 環境構築 (Installation)

ローカル環境で実行する場合は、以下のコマンドでリポジトリをクローンし、必要なライブラリをインストールしてください。

```bash

git clone [https://github.com/yuya0305-kenkyu/gw-classification.git](https://github.com/yuya0305-kenkyu/gw-classification.git)
cd gw-classification
pip install -r requirements.txt

```
Note: Google Colabを利用する場合は、notebooks/Tutorial_Colab.ipynb をColabにアップロードして実行するだけで、環境構築から推論までの一連のフローをブラウザ上で体験できます。

## 使い方 (Usage)
GitHubのファイル容量制限（100MB）を回避するため、重力波波形データ（HDF5ファイル）はリポジトリに含まれていません。実行前にローカルでデータを生成する必要があります。

### 1. データセットの生成
PyCBC を使用してノイズに埋もれた重力波シグナルを生成します。

```bash

# 学習用データの生成
python data_generation/generate_data.py --mode train

# 検証用・テスト用データの生成
python data_generation/generate_data.py --mode val
python data_generation/generate_data.py --mode test

```

※実行後、data_generation/ フォルダ内に training_data.hdf5 等が生成されます。

### 2. モデルの学習
生成したHDF5ファイルを使用して学習を開始します。

* `--method 1`: MLP

* `--method 2`: TCN

```bash
# TCNモデルを使用し、KAGRAデータを含めて学習する例
python train.py --hdf_file data_generation/training_data.hdf5 --method 2 --use_KAGRA
```

学習が完了すると、`weights/` ディレクトリにモデルの重み（`.pth`）と学習履歴のCSVが保存されます。

### 3. 学習履歴の確認
LossとAccuracyの推移をグラフ化して過学習等を確認します。

```bash
python plot_history.py weights/history_TCN_HLVK18.csv
```

### 4. 推論（予測）の実行
テストデータに対して予測を行い、精度（Accuracy）を出力します。

`--method 3`: MLPとTCNのアンサンブル予測

```bash
# TCNモデルで推論を行う例
python predict.py --hdf_file data_generation/test_data.hdf5 --method 2 --use_KAGRA
```

