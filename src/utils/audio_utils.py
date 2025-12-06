"""
Utilitários para processamento de áudio.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import noisereduce as nr


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Carrega arquivo de áudio.
    
    Args:
        file_path: Caminho para o arquivo de áudio
        sample_rate: Taxa de amostragem desejada
        duration: Duração do áudio a carregar (segundos)
        offset: Deslocamento inicial (segundos)
    
    Returns:
        Tuple com (audio_data, sample_rate)
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            duration=duration,
            offset=offset,
            mono=True
        )
        return audio, sr
    except Exception as e:
        raise ValueError(f"Erro ao carregar áudio {file_path}: {str(e)}")


def save_audio(
    audio: np.ndarray,
    file_path: str,
    sample_rate: int = 16000
) -> None:
    """
    Salva array de áudio em arquivo.
    
    Args:
        audio: Array numpy com dados de áudio
        file_path: Caminho para salvar o arquivo
        sample_rate: Taxa de amostragem
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(file_path, audio, sample_rate)
    except Exception as e:
        raise ValueError(f"Erro ao salvar áudio {file_path}: {str(e)}")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normaliza áudio para range [-1, 1].
    
    Args:
        audio: Array numpy com dados de áudio
    
    Returns:
        Áudio normalizado
    """
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio


def remove_noise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    stationary: bool = True
) -> np.ndarray:
    """
    Remove ruído do áudio usando spectral gating.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        stationary: Se True, assume ruído estacionário
    
    Returns:
        Áudio com ruído reduzido
    """
    try:
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=0.8
        )
        return reduced_noise
    except Exception as e:
        print(f"Aviso: Não foi possível remover ruído: {str(e)}")
        return audio


def apply_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Aplica filtro de pré-ênfase para realçar altas frequências.
    
    Args:
        audio: Array numpy com dados de áudio
        coef: Coeficiente de pré-ênfase
    
    Returns:
        Áudio com pré-ênfase aplicada
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    top_db: int = 20
) -> np.ndarray:
    """
    Remove silêncio do início e fim do áudio.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        top_db: Threshold em dB abaixo do qual é considerado silêncio
    
    Returns:
        Áudio sem silêncio nas pontas
    """
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio_trimmed


def pad_audio(
    audio: np.ndarray,
    target_length: int,
    mode: str = 'constant'
) -> np.ndarray:
    """
    Adiciona padding ou trunca áudio para tamanho desejado.
    
    Args:
        audio: Array numpy com dados de áudio
        target_length: Tamanho alvo em samples
        mode: Modo de padding ('constant', 'wrap', 'edge')
    
    Returns:
        Áudio com tamanho ajustado
    """
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        pad_width = target_length - len(audio)
        return np.pad(audio, (0, pad_width), mode=mode)
    return audio


def split_audio_segments(
    audio: np.ndarray,
    sample_rate: int = 16000,
    segment_duration: float = 2.0,
    overlap: float = 0.5
) -> list:
    """
    Divide áudio em segmentos com sobreposição.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        segment_duration: Duração de cada segmento (segundos)
        overlap: Sobreposição entre segmentos (0-1)
    
    Returns:
        Lista de segmentos de áudio
    """
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
    
    return segments


def get_audio_duration(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Calcula duração do áudio em segundos.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
    
    Returns:
        Duração em segundos
    """
    return len(audio) / sample_rate


def detect_cough_events(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.3,
    min_duration: float = 0.2,
    max_duration: float = 2.0
) -> list:
    """
    Detecta eventos de tosse no áudio.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        threshold: Threshold de energia para detecção
        min_duration: Duração mínima de tosse (segundos)
        max_duration: Duração máxima de tosse (segundos)
    
    Returns:
        Lista de tuplas (start_time, end_time) dos eventos detectados
    """
    # Calcula energia do sinal
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    energy = np.array([
        sum(abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio)-frame_length, hop_length)
    ])
    
    # Normaliza energia
    energy = energy / np.max(energy) if np.max(energy) > 0 else energy
    
    # Detecta regiões acima do threshold
    above_threshold = energy > threshold
    
    events = []
    in_event = False
    start_frame = 0
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_event:
            start_frame = i
            in_event = True
        elif not is_above and in_event:
            duration = (i - start_frame) * hop_length / sample_rate
            if min_duration <= duration <= max_duration:
                start_time = start_frame * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                events.append((start_time, end_time))
            in_event = False
    
    return events

