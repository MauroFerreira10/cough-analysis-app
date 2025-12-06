"""
Script para extrair características de áudio.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import pickle
import pandas as pd

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.audio_utils import load_audio


class FeatureExtractor:
    """Classe para extrair características de áudio."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o extrator de características.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.n_mfcc = self.config['audio']['n_mfcc']
        self.n_fft = self.config['audio']['n_fft']
        self.hop_length = self.config['audio']['hop_length']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio']['fmin']
        self.fmax = self.config['audio']['fmax']
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai coeficientes MFCC.
        
        Args:
            audio: Array numpy com dados de áudio
        
        Returns:
            Array com coeficientes MFCC
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Calcula estatísticas dos MFCCs
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
        mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std, mfcc_delta, mfcc_delta2])
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai espectrograma Mel.
        
        Args:
            audio: Array numpy com dados de áudio
        
        Returns:
            Espectrograma Mel normalizado
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        # Converte para dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normaliza para [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai características chroma.
        
        Args:
            audio: Array numpy com dados de áudio
        
        Returns:
            Array com características chroma
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        return np.concatenate([chroma_mean, chroma_std])
    
    def extract_zcr(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai Zero Crossing Rate.
        
        Args:
            audio: Array numpy com dados de áudio
        
        Returns:
            Array com estatísticas de ZCR
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        return np.array([zcr_mean, zcr_std])
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai características espectrais.
        
        Args:
            audio: Array numpy com dados de áudio
        
        Returns:
            Array com características espectrais
        """
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        features = np.concatenate([
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            np.mean(spectral_contrast, axis=1),
            np.std(spectral_contrast, axis=1)
        ])
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai características temporais.
        
        Args:
            audio: Array numpy com dados de áudio
        
        Returns:
            Array com características temporais
        """
        # RMS Energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Tempo
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)
        
        features = np.concatenate([
            [np.mean(rms), np.std(rms), np.max(rms)],
            tempo
        ])
        
        return features
    
    def extract_all_features(self, audio: np.ndarray) -> dict:
        """
        Extrai todas as características.
        
        Args:
            audio: Array numpy com dados de áudio
        
        Returns:
            Dicionário com todas as características
        """
        features = {
            'mfcc': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'chroma': self.extract_chroma(audio),
            'zcr': self.extract_zcr(audio),
            'spectral': self.extract_spectral_features(audio),
            'temporal': self.extract_temporal_features(audio)
        }
        
        return features
    
    def process_file(self, file_path: str, label: str = None) -> dict:
        """
        Processa um arquivo e extrai características.
        
        Args:
            file_path: Caminho do arquivo de áudio
            label: Label/classe do áudio (opcional)
        
        Returns:
            Dicionário com características e metadata
        """
        try:
            # Carrega áudio
            audio, sr = load_audio(file_path, sample_rate=self.sample_rate)
            
            # Extrai características
            features = self.extract_all_features(audio)
            
            # Concatena features vetoriais (exceto mel_spectrogram)
            feature_vector = np.concatenate([
                features['mfcc'],
                features['chroma'],
                features['zcr'],
                features['spectral'],
                features['temporal']
            ])
            
            result = {
                'file_path': file_path,
                'label': label,
                'feature_vector': feature_vector,
                'mel_spectrogram': features['mel_spectrogram'],
                'mfcc_full': features['mfcc']
            }
            
            return result
            
        except Exception as e:
            print(f"Erro ao processar {file_path}: {str(e)}")
            return None
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        label_mapping: dict = None
    ) -> None:
        """
        Processa diretório e extrai características.
        
        Args:
            input_dir: Diretório com áudios
            output_dir: Diretório para salvar características
            label_mapping: Dicionário mapeando subdiretórios para labels
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Busca arquivos de áudio
        audio_files = list(input_path.rglob('*.wav'))
        
        print(f"Encontrados {len(audio_files)} arquivos de áudio")
        
        # Processa cada arquivo
        all_features = []
        all_labels = []
        all_mel_specs = []
        metadata = []
        
        for audio_file in tqdm(audio_files, desc="Extraindo características"):
            # Determina label baseado na estrutura de diretórios
            label = None
            if label_mapping:
                for subfolder, lbl in label_mapping.items():
                    if subfolder in str(audio_file):
                        label = lbl
                        break
            
            result = self.process_file(str(audio_file), label)
            
            if result:
                all_features.append(result['feature_vector'])
                all_labels.append(result['label'])
                all_mel_specs.append(result['mel_spectrogram'])
                metadata.append({
                    'file_path': result['file_path'],
                    'label': result['label']
                })
        
        # Salva características
        features_data = {
            'features': np.array(all_features),
            'labels': all_labels,
            'mel_spectrograms': all_mel_specs,
            'metadata': metadata
        }
        
        with open(output_path / 'features.pkl', 'wb') as f:
            pickle.dump(features_data, f)
        
        # Salva metadata em CSV
        df = pd.DataFrame(metadata)
        df.to_csv(output_path / 'features_metadata.csv', index=False)
        
        print(f"\nCaracterísticas extraídas e salvas em {output_path}")
        print(f"Shape das features: {features_data['features'].shape}")
        print(f"Número de mel spectrograms: {len(all_mel_specs)}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Extrai características de arquivos de áudio'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Diretório com áudios processados'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Diretório para salvar características'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração'
    )
    parser.add_argument(
        '--labels',
        type=str,
        help='Mapeamento de labels (formato: folder1:label1,folder2:label2)'
    )
    
    args = parser.parse_args()
    
    # Parse label mapping
    label_mapping = None
    if args.labels:
        label_mapping = {}
        for pair in args.labels.split(','):
            folder, label = pair.split(':')
            label_mapping[folder] = label
    
    # Inicializa extrator
    extractor = FeatureExtractor(config_path=args.config)
    
    # Processa diretório
    extractor.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        label_mapping=label_mapping
    )


if __name__ == '__main__':
    main()

