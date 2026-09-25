gw-classification/
├── data_generation/
│   ├── generate_data.py    # 3つの生成コードを統合し、引数でtrain/test等を切り替えるスクリプト
│   └── sensitivities/      # (※ユーザーが配置するO4やKAGRAの感度ファイル群)
├── models/
│   ├── __init__.py
│   ├── mlp.py              # MLPクラスの定義
│   └── tcn.py              # TemporalBlock, TemporalConvNet, TCNクラスの定義
├── utils/
│   ├── __init__.py
│   ├── dataset.py          # AugmentedDataset, 各種DataLoaderの生成関数
│   └── preprocess.py       # データフレーム化、特徴量抽出など (元の preprocess.py の一部)
├── train.py                # 引数を解釈し、モデルの学習を実行するメインスクリプト
├── predict.py              # 学習済み重みを読み込み、推論を行うスクリプト
├── plot_history.py         # 学習曲線の描画スクリプト
├── requirements.txt        # 必要なライブラリ一覧 (PyCBC, torch, torchaudio, etc.)
├── README.md               # 実行手順を記載したドキュメント
└── notebooks/
    └── Tutorial_Colab.ipynb # Colab上で一連の処理を実行するためのノートブック
