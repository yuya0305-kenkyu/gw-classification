import argparse
import os
import h5py
import numpy as np
import pycbc.detector
import pycbc.noise
import pycbc.psd
from pycbc.filter import sigmasq
from pycbc.waveform import get_td_waveform
from tqdm import tqdm

PRESETS = {
    'sample': {'seed': 1, 'num_samples': 100, 'output_file': 'sample_data.hdf5'},
    'test': {'seed': 9999, 'num_samples': 4000, 'output_file': 'test_data.hdf5'},
    'train': {'seed': 42, 'num_samples': 20000, 'output_file': 'training_data.hdf5'},
    'val': {'seed': 1234, 'num_samples': 4000, 'output_file': 'validation_data.hdf5'}
}

BASE_CONFIG = {
    'mass_range': [30.0, 80.0],
    'spin_range': [0.0, 0.998],
    'snr_range': [10.0, 50.0],
    'duration': 32.0,
    'sample_rate': 2048,
    'f_lower': 20.0,
    'approximant': 'IMRPhenomPv2',
    'slice_start': -0.20,
    'slice_end': 0.05
}

PSD_FILES = {
    'H1': 'sensitivities/aligo_O4high.txt',
    'L1': 'sensitivities/aligo_O4high.txt',
    'V1': 'sensitivities/avirgo_O4high_NEW.txt',
    'K1': 'sensitivities/kagra_128Mpc.txt'
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic gravitational wave datasets for training/testing.")
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'sample', 'val'], default='sample',
                        help="Data generation mode preset (train, test, sample, val)")
    parser.add_argument("--num_samples", type=int, default=None, help="Override sample count")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output_file", type=str, default=None, help="Override output HDF5 path")
    parser.add_argument("--sensitivities_dir", type=str, default="sensitivities", help="Directory of PSD text files")
    return parser.parse_args()


def sample_parameters(config: dict) -> dict:
    m1 = np.random.uniform(*config['mass_range'])
    m2 = np.random.uniform(*config['mass_range'])
    mass1, mass2 = max(m1, m2), min(m1, m2)

    spin1z = np.random.uniform(*config['spin_range'])
    spin2z = np.random.uniform(*config['spin_range'])

    ra = np.random.uniform(0, 2 * np.pi)
    dec = np.arcsin(np.random.uniform(-1, 1))
    pol = np.random.uniform(0, 2 * np.pi)
    inc = np.arccos(np.random.uniform(-1, 1))
    coa_phase = np.random.uniform(0, 2 * np.pi)
    target_snr = np.random.uniform(*config['snr_range'])

    return {
        'mass1': mass1, 'mass2': mass2, 'spin1z': spin1z, 'spin2z': spin2z,
        'ra': ra, 'dec': dec, 'inclination': inc, 'polarization': pol,
        'coa_phase': coa_phase, 'target_snr': target_snr
    }


def load_psd(det_name: str, flen: int, delta_f: float, f_lower: float, sensitivities_dir: str):
    path = os.path.join(sensitivities_dir, os.path.basename(PSD_FILES.get(det_name, '')))
    try:
        psd = pycbc.psd.from_txt(path, flen, delta_f, f_lower, is_asd_file=True)
    except Exception:
        # 感度ファイルが存在しない場合は解析的PSDモデルへフォールバック
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_lower)
        if det_name == 'K1':
            psd.data *= 10.0
    return psd


def generate_dataset(mode: str, num_samples: int, seed: int, output_file: str, sensitivities_dir: str):
    config = BASE_CONFIG.copy()
    np.random.seed(seed)

    sample_rate = config['sample_rate']
    slice_points = int(round((config['slice_end'] - config['slice_start']) * sample_rate))

    print(f"[{mode.upper()} MODE] Generating {num_samples} samples with Seed {seed} -> {output_file}")

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    data_buffer = {
        'h1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'l1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'v1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'k1_strain': np.zeros((num_samples, slice_points), dtype='float32')
    }

    param_buffer = {
        k: np.zeros(num_samples, dtype='float32')
        for k in ['mass1', 'mass2', 'spin1z', 'spin2z', 'ra', 'dec', 'injection_snr']
    }

    detectors = {name: pycbc.detector.Detector(name) for name in ['H1', 'L1', 'V1', 'K1']}
    det_indices = {'H1': 0, 'L1': 1, 'V1': 2, 'K1': 3}

    for i in tqdm(range(num_samples), desc=f"Generating {mode} data"):
        p = sample_parameters(config)

        # 1. 基準波形生成
        hp, hc = get_td_waveform(
            approximant=config['approximant'],
            mass1=p['mass1'], mass2=p['mass2'],
            spin1z=p['spin1z'], spin2z=p['spin2z'],
            distance=1000.0,
            inclination=p['inclination'],
            coa_phase=p['coa_phase'],
            delta_t=1.0 / sample_rate,
            f_lower=config['f_lower']
        )

        net_snr_sq = 0.0
        signals_no_noise = {}

        # 2. アンテナパターン投影 & 最適SNR計算
        for name, det in detectors.items():
            fp, fc = det.antenna_pattern(p['ra'], p['dec'], p['polarization'], 0.0)
            dt = det.time_delay_from_earth_center(p['ra'], p['dec'], 0.0)

            sig = fp * hp + fc * hc
            sig.start_time += dt

            sig_snr = sig.copy()
            sig_snr.resize(int(config['duration'] * sample_rate))
            flen = int(len(sig_snr) / 2) + 1
            psd = load_psd(name, flen, sig_snr.delta_f, config['f_lower'], sensitivities_dir)

            snr_sq = sigmasq(sig_snr, psd=psd, low_frequency_cutoff=config['f_lower'])
            net_snr_sq += snr_sq
            signals_no_noise[name] = (sig, psd)

        current_net_snr = np.sqrt(net_snr_sq)
        if current_net_snr == 0:
            current_net_snr = 1e-10
        scaling_factor = current_net_snr / p['target_snr']

        # 3. ノイズ合成 & 切り出し
        for name, (sig, psd) in signals_no_noise.items():
            sig_final = sig / scaling_factor

            noise_seed = seed + (i * 100) + det_indices[name]
            noise_len = int(config['duration'] * sample_rate)
            noise = pycbc.noise.noise_from_psd(noise_len, sig_final.delta_t, psd, seed=noise_seed)
            noise.start_time = -(config['duration'] / 2.0)

            strain_data = noise.numpy().copy()
            start_idx = int(round((float(sig_final.start_time) - float(noise.start_time)) * sample_rate))
            end_idx = start_idx + len(sig_final)

            slice_start = max(0, start_idx)
            slice_end = min(len(strain_data), end_idx)
            sig_start = max(0, -start_idx)
            sig_end = sig_start + (slice_end - slice_start)

            if slice_end > slice_start:
                strain_data[slice_start:slice_end] += sig_final.numpy()[sig_start:sig_end]

            ext_start_time = config['slice_start']
            ext_start_idx = int(round((float(ext_start_time) - float(noise.start_time)) * sample_rate))
            ext_end_idx = ext_start_idx + slice_points

            data_slice = strain_data[ext_start_idx:ext_end_idx]
            if len(data_slice) < slice_points:
                data_slice = np.pad(data_slice, (0, slice_points - len(data_slice)))

            key_name = name.lower() + '_strain'
            data_buffer[key_name][i] = data_slice

        for k in ['mass1', 'mass2', 'spin1z', 'spin2z', 'ra', 'dec']:
            param_buffer[k][i] = p[k]
        param_buffer['injection_snr'][i] = p['target_snr']

    with h5py.File(output_file, 'w') as f:
        grp_samples = f.create_group('injection_samples')
        grp_params = f.create_group('injection_parameters')
        for key, val in data_buffer.items():
            grp_samples.create_dataset(key, data=val)
        for key, val in param_buffer.items():
            grp_params.create_dataset(key, data=val)

    print(f"Successfully generated: {output_file}")


if __name__ == "__main__":
    args = parse_arguments()
    preset = PRESETS[args.mode]

    num_samples = args.num_samples if args.num_samples is not None else preset['num_samples']
    seed = args.seed if args.seed is not None else preset['seed']
    output_file = args.output_file if args.output_file is not None else preset['output_file']

    generate_dataset(args.mode, num_samples, seed, output_file, args.sensitivities_dir)