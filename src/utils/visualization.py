"""
Utilitários para visualização de dados de áudio.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, Tuple


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Forma de Onda",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plota forma de onda do áudio.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar a figura (opcional)
    """
    plt.figure(figsize=figsize)
    librosa.display.waveshow(audio, sr=sample_rate, alpha=0.8)
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Espectrograma",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plota espectrograma do áudio.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        n_fft: Tamanho da FFT
        hop_length: Hop length para STFT
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar a figura (opcional)
    """
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    title: str = "Espectrograma Mel",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plota espectrograma Mel do áudio.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        n_fft: Tamanho da FFT
        hop_length: Hop length para STFT
        n_mels: Número de bandas Mel
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar a figura (opcional)
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_mfcc(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "MFCC",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plota coeficientes MFCC.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        n_mfcc: Número de coeficientes MFCC
        n_fft: Tamanho da FFT
        hop_length: Hop length para STFT
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar a figura (opcional)
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        mfccs,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        cmap='viridis'
    )
    plt.colorbar()
    plt.title(title)
    plt.ylabel('MFCC Coeficientes')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_multiple_spectrograms(
    audios: list,
    labels: list,
    sample_rate: int = 16000,
    n_mels: int = 128,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plota múltiplos espectrogramas para comparação.
    
    Args:
        audios: Lista de arrays de áudio
        labels: Lista de labels para cada áudio
        sample_rate: Taxa de amostragem
        n_mels: Número de bandas Mel
        figsize: Tamanho da figura
        save_path: Caminho para salvar a figura (opcional)
    """
    n_plots = len(audios)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (audio, label) in enumerate(zip(audios, labels)):
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        
        librosa.display.specshow(
            S_db,
            sr=sample_rate,
            x_axis='time',
            y_axis='mel',
            cmap='viridis',
            ax=axes[i]
        )
        axes[i].set_title(label)
        axes[i].label_outer()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_feature_comparison(
    features_dict: dict,
    title: str = "Comparação de Características",
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plota comparação entre diferentes características extraídas.
    
    Args:
        features_dict: Dicionário com nome da feature como chave e valores como array
        title: Título do gráfico
        figsize: Tamanho da figura
        save_path: Caminho para salvar a figura (opcional)
    """
    n_features = len(features_dict)
    fig, axes = plt.subplots(n_features, 1, figsize=figsize)
    
    if n_features == 1:
        axes = [axes]
    
    for i, (feature_name, feature_values) in enumerate(features_dict.items()):
        if len(feature_values.shape) == 1:
            axes[i].plot(feature_values)
        else:
            im = axes[i].imshow(
                feature_values,
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            plt.colorbar(im, ax=axes[i])
        
        axes[i].set_title(feature_name)
        axes[i].set_xlabel('Frame')
    
    plt.suptitle(title, fontsize=16, y=1.001)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

