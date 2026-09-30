import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import healpy as hp
import numpy as np
import glob
import os

# Seabornのスタイル設定でグラフを美しくする
sns.set_theme(style="whitegrid")

def plot_snr_accuracy(df_list, output_dir):
    """SNR帯ごとの正解率を折れ線グラフで可視化"""
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # SNRを5刻みのビン（階級）に分割
    bins = np.arange(10, 55, 5)
    combined_df['SNR_Bin'] = pd.cut(combined_df['SNR'], bins=bins, right=False)
    
    # ネットワーク(HLV/HLVK)とSNRビンでグループ化して正解率を計算
    acc_df = combined_df.groupby(['Network', 'SNR_Bin'], observed=False)['Is_Correct'].mean().reset_index()
    acc_df['SNR_Center'] = acc_df['SNR_Bin'].apply(lambda x: x.mid).astype(float)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=acc_df, x='SNR_Center', y='Is_Correct', hue='Network', marker='o', linewidth=2, markersize=8)
    
    plt.title('Accuracy by SNR Band (TCN)', fontsize=16)
    plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(title='Detector Network', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_snr_accuracy.png'), dpi=300)
    plt.close()
    print("Saved: plot_snr_accuracy.png (SNRごとの正解率S字カーブ)")

def plot_error_distance(df, output_dir):
    """間違えた予測が正解セクタから物理的にどれくらい離れていたか（角度）のヒストグラム"""
    # 不正解だったデータのみを抽出
    errors = df[df['Is_Correct'] == 0].copy()
    if len(errors) == 0:
        return
        
    n_sectors = df['N_Sectors'].iloc[0]
    network = df['Network'].iloc[0]
    nside = int((n_sectors / 12) ** 0.5)
    
    # HEALPixのピクセルインデックスから3次元方向ベクトルを取得
    vec_true = np.array(hp.pix2vec(nside, errors['True_Sector'].values, nest=True))
    vec_pred = np.array(hp.pix2vec(nside, errors['Pred_Sector'].values, nest=True))
    
    # 内積を用いて2つのベクトル（正解方向と予測方向）のなす角（ラジアン）を計算
    dot_product = np.sum(vec_true * vec_pred, axis=0)
    dot_product = np.clip(dot_product, -1.0, 1.0) # 演算誤差によるNaN防止
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad) # 度数法に変換
    
    plt.figure(figsize=(10, 6))
    sns.histplot(angle_deg, bins=30, kde=True, color='indianred')
    plt.title(f'Angular Error for Incorrect Predictions ({network}, {n_sectors} Sectors)', fontsize=16)
    plt.xlabel('Distance from True Sector (Degrees)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    
    # 補足テキストの追加（中央値など）
    median_error = np.median(angle_deg)
    plt.axvline(median_error, color='black', linestyle='--', label=f'Median Error: {median_error:.1f}°')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    out_name = f'plot_error_distance_{network}_{n_sectors}.png'
    plt.savefig(os.path.join(output_dir, out_name), dpi=300)
    plt.close()
    print(f"Saved: {out_name} (不正解データの「惜しさ」の分布)")

if __name__ == "__main__":
    # カレントディレクトリにある "results_*.csv" をすべて読み込む
    csv_files = glob.glob("results_*.csv")
    
    if not csv_files:
        print("CSVファイルが見つかりません。先に predict.py を実行してCSVを生成してください。")
    else:
        output_dir = "plots_output"
        os.makedirs(output_dir, exist_ok=True)
        
        df_list = [pd.read_csv(f) for f in csv_files]
        
        # 1. 全CSVデータを統合したSNR別比較グラフの作成
        plot_snr_accuracy(df_list, output_dir)
        
        # 2. 各CSV（解像度・ネットワーク別）の誤差距離ヒストグラムの作成
        for df in df_list:
            plot_error_distance(df, output_dir)
            
        print(f"\nすべてのグラフを '{output_dir}/' フォルダに出力しました。")