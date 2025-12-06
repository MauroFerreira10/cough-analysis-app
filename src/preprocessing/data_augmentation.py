"""
Script para data augmentation de áudio.
"""

import numpy as np
import librosa
from typing import Tuple


class AudioAugmenter:
    """Classe para aplicar data augmentation em áudio."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Inicializa o augmentador.
        
        Args:
            sample_rate: Taxa de amostragem
        """
        self.sample_rate = sample_rate
    
    def time_stretch(
        self,
        audio: np.ndarray,
        rate: float = 1.0
    ) -> np.ndarray:
        """
        Aplica time stretching (altera velocidade sem mudar pitch).
        
        Args:
            audio: Array numpy com dados de áudio
            rate: Taxa de stretching (< 1 = mais lento, > 1 = mais rápido)
        
        Returns:
            Áudio com time stretching aplicado
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: float = 0.0
    ) -> np.ndarray:
        """
        Altera o pitch do áudio.
        
        Args:
            audio: Array numpy com dados de áudio
            n_steps: Número de semitons para alterar (+ ou -)
        
        Returns:
            Áudio com pitch alterado
        """
        return librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=n_steps
        )
    
    def add_noise(
        self,
        audio: np.ndarray,
        noise_factor: float = 0.005
    ) -> np.ndarray:
        """
        Adiciona ruído gaussiano ao áudio.
        
        Args:
            audio: Array numpy com dados de áudio
            noise_factor: Fator de ruído (amplitude)
        
        Returns:
            Áudio com ruído adicionado
        """
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        # Normaliza para evitar clipping
        return augmented / np.max(np.abs(augmented))
    
    def time_shift(
        self,
        audio: np.ndarray,
        shift_max: float = 0.2
    ) -> np.ndarray:
        """
        Desloca o áudio no tempo.
        
        Args:
            audio: Array numpy com dados de áudio
            shift_max: Deslocamento máximo (fração do comprimento total)
        
        Returns:
            Áudio deslocado
        """
        shift = np.random.randint(int(len(audio) * shift_max))
        direction = np.random.choice([-1, 1])
        shift = shift * direction
        return np.roll(audio, shift)
    
    def change_volume(
        self,
        audio: np.ndarray,
        factor_range: Tuple[float, float] = (0.7, 1.3)
    ) -> np.ndarray:
        """
        Altera o volume do áudio.
        
        Args:
            audio: Array numpy com dados de áudio
            factor_range: Range de fator de volume (min, max)
        
        Returns:
            Áudio com volume alterado
        """
        factor = np.random.uniform(factor_range[0], factor_range[1])
        augmented = audio * factor
        # Clipping
        augmented = np.clip(augmented, -1.0, 1.0)
        return augmented
    
    def random_augment(
        self,
        audio: np.ndarray,
        n_augmentations: int = 1
    ) -> list:
        """
        Aplica augmentações aleatórias.
        
        Args:
            audio: Array numpy com dados de áudio
            n_augmentations: Número de versões augmentadas a gerar
        
        Returns:
            Lista de áudios augmentados
        """
        augmented_audios = []
        
        augmentation_methods = [
            lambda x: self.time_stretch(x, rate=np.random.uniform(0.8, 1.2)),
            lambda x: self.pitch_shift(x, n_steps=np.random.randint(-2, 3)),
            lambda x: self.add_noise(x, noise_factor=np.random.uniform(0.001, 0.01)),
            lambda x: self.time_shift(x, shift_max=0.2),
            lambda x: self.change_volume(x, factor_range=(0.7, 1.3))
        ]
        
        for _ in range(n_augmentations):
            # Seleciona aleatoriamente 1-3 augmentações para aplicar
            n_methods = np.random.randint(1, 4)
            selected_methods = np.random.choice(
                augmentation_methods,
                size=n_methods,
                replace=False
            )
            
            # Aplica augmentações sequencialmente
            augmented = audio.copy()
            for method in selected_methods:
                try:
                    augmented = method(augmented)
                except Exception as e:
                    print(f"Erro ao aplicar augmentação: {str(e)}")
                    continue
            
            augmented_audios.append(augmented)
        
        return augmented_audios
    
    def augment_batch(
        self,
        audios: list,
        n_augmentations: int = 2
    ) -> Tuple[list, list]:
        """
        Aplica augmentação em um batch de áudios.
        
        Args:
            audios: Lista de áudios
            n_augmentations: Número de augmentações por áudio
        
        Returns:
            Tupla (áudios originais + augmentados, índices originais)
        """
        all_audios = []
        indices = []
        
        for i, audio in enumerate(audios):
            # Adiciona áudio original
            all_audios.append(audio)
            indices.append(i)
            
            # Adiciona versões augmentadas
            augmented = self.random_augment(audio, n_augmentations)
            all_audios.extend(augmented)
            indices.extend([i] * len(augmented))
        
        return all_audios, indices


def apply_augmentation_pipeline(
    audio: np.ndarray,
    sample_rate: int = 16000,
    config: dict = None
) -> list:
    """
    Aplica pipeline de augmentação baseado em configuração.
    
    Args:
        audio: Array numpy com dados de áudio
        sample_rate: Taxa de amostragem
        config: Dicionário de configuração com parâmetros de augmentação
    
    Returns:
        Lista de áudios augmentados
    """
    if config is None:
        config = {
            'time_stretch_rate': [0.8, 1.2],
            'pitch_shift_steps': [-2, 2],
            'noise_factor': 0.005,
            'shift_max': 0.2
        }
    
    augmenter = AudioAugmenter(sample_rate=sample_rate)
    
    augmented_audios = []
    
    # Time stretch
    rate = np.random.uniform(
        config['time_stretch_rate'][0],
        config['time_stretch_rate'][1]
    )
    augmented_audios.append(augmenter.time_stretch(audio, rate=rate))
    
    # Pitch shift
    n_steps = np.random.randint(
        config['pitch_shift_steps'][0],
        config['pitch_shift_steps'][1] + 1
    )
    augmented_audios.append(augmenter.pitch_shift(audio, n_steps=n_steps))
    
    # Add noise
    augmented_audios.append(
        augmenter.add_noise(audio, noise_factor=config['noise_factor'])
    )
    
    # Time shift
    augmented_audios.append(
        augmenter.time_shift(audio, shift_max=config['shift_max'])
    )
    
    # Combinação de augmentações
    combined = augmenter.time_stretch(audio, rate=0.9)
    combined = augmenter.add_noise(combined, noise_factor=0.003)
    augmented_audios.append(combined)
    
    return augmented_audios

