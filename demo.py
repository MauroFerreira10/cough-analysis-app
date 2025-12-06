#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de demonstração do sistema de análise de tosse.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Adiciona diretório raiz ao path
sys.path.append(str(Path(__file__).parent))

from src.utils.audio_utils import load_audio, normalize_audio, remove_noise
from src.preprocessing.extract_features import FeatureExtractor
from src.utils.visualization import (
    plot_waveform,
    plot_mel_spectrogram,
    plot_mfcc
)


def demo_audio_loading(audio_path: str):
    """Demonstra carregamento de áudio."""
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Carregamento de Áudio")
    print("="*60)
    
    try:
        audio, sr = load_audio(audio_path, sample_rate=16000)
        print(f"✓ Áudio carregado com sucesso")
        print(f"  Taxa de amostragem: {sr} Hz")
        print(f"  Duração: {len(audio)/sr:.2f} segundos")
        print(f"  Número de samples: {len(audio)}")
        print(f"  Amplitude: min={audio.min():.3f}, max={audio.max():.3f}")
        return audio, sr
    except Exception as e:
        print(f"✗ Erro ao carregar áudio: {e}")
        return None, None


def demo_preprocessing(audio: np.ndarray, sr: int):
    """Demonstra pré-processamento de áudio."""
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Pré-processamento")
    print("="*60)
    
    # Normalização
    print("\n1. Normalizando áudio...")
    audio_normalized = normalize_audio(audio)
    print(f"   Amplitude após normalização: min={audio_normalized.min():.3f}, "
          f"max={audio_normalized.max():.3f}")
    
    # Remoção de ruído
    print("\n2. Removendo ruído...")
    audio_denoised = remove_noise(audio_normalized, sample_rate=sr)
    print(f"   ✓ Ruído removido")
    
    return audio_denoised


def demo_feature_extraction(audio: np.ndarray):
    """Demonstra extração de características."""
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Extração de Características")
    print("="*60)
    
    extractor = FeatureExtractor()
    
    print("\n1. Extraindo MFCC...")
    mfcc = extractor.extract_mfcc(audio)
    print(f"   Shape: {mfcc.shape}")
    print(f"   Range: [{mfcc.min():.2f}, {mfcc.max():.2f}]")
    
    print("\n2. Extraindo Mel Spectrogram...")
    mel_spec = extractor.extract_mel_spectrogram(audio)
    print(f"   Shape: {mel_spec.shape}")
    print(f"   Range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")
    
    print("\n3. Extraindo Zero Crossing Rate...")
    zcr = extractor.extract_zcr(audio)
    print(f"   Mean ZCR: {zcr[0]:.4f}")
    print(f"   Std ZCR: {zcr[1]:.4f}")
    
    print("\n4. Extraindo características espectrais...")
    spectral = extractor.extract_spectral_features(audio)
    print(f"   Número de features: {len(spectral)}")
    
    print("\n5. Extraindo características temporais...")
    temporal = extractor.extract_temporal_features(audio)
    print(f"   Número de features: {len(temporal)}")
    
    return {
        'mfcc': mfcc,
        'mel_spectrogram': mel_spec,
        'zcr': zcr,
        'spectral': spectral,
        'temporal': temporal
    }


def demo_visualization(audio: np.ndarray, sr: int, output_dir: str = None):
    """Demonstra visualizações."""
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Visualizações")
    print("="*60)
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n1. Plotando forma de onda...")
    save_path = f"{output_dir}/waveform.png" if output_dir else None
    plot_waveform(audio, sr, title="Forma de Onda", save_path=save_path)
    if save_path:
        print(f"   Salvo em: {save_path}")
    
    print("\n2. Plotando espectrograma Mel...")
    save_path = f"{output_dir}/mel_spectrogram.png" if output_dir else None
    plot_mel_spectrogram(audio, sr, title="Espectrograma Mel", save_path=save_path)
    if save_path:
        print(f"   Salvo em: {save_path}")
    
    print("\n3. Plotando MFCC...")
    save_path = f"{output_dir}/mfcc.png" if output_dir else None
    plot_mfcc(audio, sr, title="MFCC", save_path=save_path)
    if save_path:
        print(f"   Salvo em: {save_path}")


def demo_full_pipeline(audio_path: str, output_dir: str = None):
    """Executa pipeline completo de demonstração."""
    print("\n" + "="*70)
    print(" "*15 + "DEMONSTRAÇÃO DO SISTEMA DE ANÁLISE DE TOSSE")
    print("="*70)
    
    # 1. Carregamento
    audio, sr = demo_audio_loading(audio_path)
    if audio is None:
        return
    
    # 2. Pré-processamento
    audio_processed = demo_preprocessing(audio, sr)
    
    # 3. Extração de características
    features = demo_feature_extraction(audio_processed)
    
    # 4. Visualização
    if output_dir or '--no-viz' not in sys.argv:
        demo_visualization(audio_processed, sr, output_dir)
    
    print("\n" + "="*70)
    print("✓ Demonstração concluída com sucesso!")
    print("="*70 + "\n")


def create_synthetic_audio():
    """Cria áudio sintético para demonstração."""
    print("\nCriando áudio sintético para demonstração...")
    
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simula tosse: explosão inicial + decaimento
    frequency = 500  # Hz
    audio = np.sin(2 * np.pi * frequency * t) * np.exp(-3 * t)
    
    # Adiciona ruído
    noise = np.random.randn(len(audio)) * 0.05
    audio = audio + noise
    
    # Normaliza
    audio = audio / np.max(np.abs(audio))
    
    return audio, sr


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Demonstração do Sistema de Análise de Tosse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python demo.py --audio caminho/para/tosse.wav
  python demo.py --audio caminho/para/tosse.wav --output demos/
  python demo.py --synthetic
        """
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        help='Caminho para arquivo de áudio de tosse'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Diretório para salvar visualizações'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Usar áudio sintético para demonstração'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Desabilita visualizações'
    )
    
    args = parser.parse_args()
    
    # Usa áudio sintético se nenhum arquivo fornecido
    if args.synthetic or not args.audio:
        print("Modo de demonstração com áudio sintético")
        audio, sr = create_synthetic_audio()
        
        # Salva temporariamente
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            import soundfile as sf
            sf.write(f.name, audio, sr)
            audio_path = f.name
        
        demo_full_pipeline(audio_path, args.output)
        
        # Remove arquivo temporário
        Path(audio_path).unlink()
    else:
        demo_full_pipeline(args.audio, args.output)


if __name__ == '__main__':
    main()

