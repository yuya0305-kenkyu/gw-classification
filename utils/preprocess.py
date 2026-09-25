import random
import h5py
import healpy as hp
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import torch


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_dataframe(hdf_paths: list) -> pd.DataFrame:
    """
    HDF5ファイル群から各干渉計間の相互相関・遅延・振幅比・位相差等の特徴量を抽出してDataFrameを作成。
    """
    M, dec, ra, snr = np.zeros((0, 42)), [], [], []
    for hdf_path in hdf_paths:
        with h5py.File(hdf_path, 'r') as f:
            h1_strain = f['injection_samples']['h1_strain'][()]
            l1_strain = f['injection_samples']['l1_strain'][()]
            v1_strain = f['injection_samples']['v1_strain'][()]
            k1_strain = f['injection_samples']['k1_strain'][()]
            dec_i = f['injection_parameters']['dec'][()]
            ra_i = f['injection_parameters']['ra'][()]
            snr_i = f['injection_parameters']['injection_snr'][()]

        dec += list(dec_i)
        ra += list(ra_i)
        snr += list(snr_i)

        length = len(h1_strain)
        strains = {0: h1_strain, 1: l1_strain, 2: v1_strain, 3: k1_strain}

        for i in range(length):
            for j in range(4):
                strains[j][i] -= np.mean(strains[j][i])

        hilbert_h1 = scipy.signal.hilbert(h1_strain)
        hilbert_l1 = scipy.signal.hilbert(l1_strain)
        hilbert_v1 = scipy.signal.hilbert(v1_strain)
        hilbert_k1 = scipy.signal.hilbert(k1_strain)
        hilberts = {0: hilbert_h1, 1: hilbert_l1, 2: hilbert_v1, 3: hilbert_k1}

        features = np.zeros((length, 42))

        for i in range(length):
            features_i = []
            for d1 in range(4):
                coal_time1 = np.argmax(np.abs(strains[d1][i]))
                amp1 = strains[d1][i][coal_time1]
                phase1 = hilberts[d1][i][coal_time1].imag

                for d2 in range(d1 + 1, 4):
                    # (i) & (ii) Signals arrival time delay and max cross-correlation
                    corr = np.correlate(strains[d1][i], strains[d2][i], 'full')
                    corr_argmax = np.argmax(abs(corr))
                    features_i.append(corr_argmax - len(strains[d1][i]) + 1)
                    features_i.append(corr[corr_argmax])

                    # (iii) & (iv) Analytic signals delay and max cross-correlation
                    corr_h = np.correlate(hilberts[d1][i], hilberts[d2][i], 'full')
                    corr_h = np.abs(corr_h)
                    corr_h_argmax = np.argmax(corr_h)
                    features_i.append(corr_h.argmax() - len(strains[d1][i]) + 1)
                    features_i.append(corr_h[corr_h_argmax])

                    # (v) & (vi) Amplitudes ratio and phase lag
                    coal_time2 = np.argmax(np.abs(strains[d2][i]))
                    amp2 = strains[d2][i][coal_time2]
                    phase2 = hilberts[d2][i][coal_time2].imag
                    features_i.append(amp1 / amp2 if amp2 != 0 else 0.0)
                    features_i.append(phase1 - phase2)

                    # (vii) Pearson Correlation Coefficient
                    ccc = scipy.stats.pearsonr(strains[d1][i], strains[d2][i])[0]
                    features_i.append(ccc if not np.isnan(ccc) else 0.0)

            features[i] = features_i

        M = np.concatenate([M, features], axis=0)

    dec, ra, snr = np.array(dec), np.array(ra), np.array(snr)

    # 標準化 (Standardization)
    if M.shape[0] > 1:
        for i in range(M.shape[1]):
            std_val = np.std(M[:, i])
            if std_val > 0:
                M[:, i] -= np.mean(M[:, i])
                M[:, i] /= std_val

    columns = [
        "Delay HL", "Max-Corr HL", "Delay-Ana HL", "Max-Corr-Ana HL",
        "Amp-Ratio HL", "Phase-lag HL", "Corr-Coef HL",
        "Delay HV", "Max-Corr HV", "Delay-Ana HV", "Max-Corr-Ana HV",
        "Amp-Ratio HV", "Phase-lag HV", "Corr-Coef HV",
        "Delay HK", "Max-Corr HK", "Delay-Ana HK", "Max-Corr-Ana HK",
        "Amp-Ratio HK", "Phase-lag HK", "Corr-Coef HK",
        "Delay LV", "Max-Corr LV", "Delay-Ana LV", "Max-Corr-Ana LV",
        "Amp-Ratio LV", "Phase-lag LV", "Corr-Coef LV",
        "Delay LK", "Max-Corr LK", "Delay-Ana LK", "Max-Corr-Ana LK",
        "Amp-Ratio LK", "Phase-lag LK", "Corr-Coef LK",
        "Delay VK", "Max-Corr VK", "Delay-Ana VK", "Max-Corr-Ana VK",
        "Amp-Ratio VK", "Phase-lag VK", "Corr-Coef VK"
    ]
    df = pd.DataFrame(M, columns=columns)
    df['dec'] = dec
    df['ra'] = ra
    df['snr'] = snr
    return df


def drop_KAGRA(df: pd.DataFrame) -> pd.DataFrame:
    """
    KAGRAに関連する特徴量列を削除して3干渉計 (H1, L1, V1) 構成にする。
    """
    columns_to_drop = [
        "Delay HK", "Delay LK", "Delay VK",
        "Max-Corr HK", "Max-Corr LK", "Max-Corr VK",
        "Delay-Ana HK", "Delay-Ana LK", "Delay-Ana VK",
        "Max-Corr-Ana HK", "Max-Corr-Ana LK", "Max-Corr-Ana VK",
        "Amp-Ratio HK", "Amp-Ratio LK", "Amp-Ratio VK",
        "Phase-lag HK", "Phase-lag LK", "Phase-lag VK",
        "Corr-Coef HK", "Corr-Coef LK", "Corr-Coef VK"
    ]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)


def create_labels_from_hdf(hdf_paths: list, n_sectors: int, HEALPix: bool = False) -> np.ndarray:
    """
    天球分割 (グリッドまたはHEALPix) に基づいてラベルを計算。
    """
    dec, ra = [], []
    for hdf_path in hdf_paths:
        with h5py.File(hdf_path, 'r') as f:
            dec += list(f['injection_parameters']['dec'][()])
            ra += list(f['injection_parameters']['ra'][()])

    dec, ra = np.array(dec), np.array(ra)

    if not HEALPix:
        delta = np.pi / int((n_sectors / 2) ** 0.5)
        dec_label = ((dec + np.pi / 2) / delta).astype(int)
        ra_label = (ra / delta).astype(int)
        labels = 2 * int((n_sectors / 2) ** 0.5) * dec_label + ra_label
    else:
        nside = int((n_sectors / 12) ** 0.5)
        labels = [hp.pixelfunc.ang2pix(nside, dec[i] + np.pi / 2, ra[i], nest=True) for i in range(len(dec))]
        labels = np.array(labels)

    return labels


def create_array_from_hdf(input_paths: list, use_KAGRA: bool) -> np.ndarray:
    """
    HDF5ファイル群から時系列歪みデータを抽出し、平均ゼロ・最大値1で正規化して (N, T, C) 形状で返却。
    """
    h1_strain, l1_strain = np.zeros((0, 512)), np.zeros((0, 512))
    v1_strain, k1_strain = np.zeros((0, 512)), np.zeros((0, 512))

    for input_path in input_paths:
        with h5py.File(input_path, 'r') as f:
            h1_strain = np.concatenate([h1_strain, f['injection_samples']['h1_strain'][()]])
            l1_strain = np.concatenate([l1_strain, f['injection_samples']['l1_strain'][()]])
            v1_strain = np.concatenate([v1_strain, f['injection_samples']['v1_strain'][()]])
            k1_strain = np.concatenate([k1_strain, f['injection_samples']['k1_strain'][()]])

    length = len(h1_strain)
    strains = {0: h1_strain, 1: l1_strain, 2: v1_strain, 3: k1_strain}

    for i in range(length):
        for j in range(4):
            strains[j][i] -= np.mean(strains[j][i])
            max_val = np.abs(strains[j][i]).max()
            if max_val > 0:
                strains[j][i] /= max_val

    if use_KAGRA:
        return np.dstack([h1_strain, l1_strain, v1_strain, k1_strain])
    else:
        return np.dstack([h1_strain, l1_strain, v1_strain])