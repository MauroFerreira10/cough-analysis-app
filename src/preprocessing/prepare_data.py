"""
Script para preparar dados de áudio brutos.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.audio_utils import (
    load_audio,
    save_audio,
    normalize_audio,
    remove_noise,
    trim_silence,
    pad_audio
)


class AudioPreprocessor:
    """Classe para pré-processar dados de áudio."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o pré-processador.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        remove_noise_flag: bool = True,
        trim_silence_flag: bool = True
    ) -> bool:
        """
        Processa um arquivo de áudio individual.
        
        Args:
            input_path: Caminho do arquivo de entrada
            output_path: Caminho do arquivo de saída
            remove_noise_flag: Se True, remove ruído
            trim_silence_flag: Se True, remove silêncio
        
        Returns:
            True se processamento bem-sucedido, False caso contrário
        """
        try:
            # Carrega áudio
            audio, sr = load_audio(input_path, sample_rate=self.sample_rate)
            
            # Remove silêncio
            if trim_silence_flag:
                audio = trim_silence(audio, sample_rate=sr)
            
            # Remove ruído
            if remove_noise_flag:
                audio = remove_noise(audio, sample_rate=sr)
            
            # Normaliza
            audio = normalize_audio(audio)
            
            # Ajusta duração
            target_length = int(self.duration * sr)
            audio = pad_audio(audio, target_length)
            
            # Salva áudio processado
            save_audio(audio, output_path, sample_rate=sr)
            
            return True
            
        except Exception as e:
            print(f"Erro ao processar {input_path}: {str(e)}")
            return False
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        remove_noise_flag: bool = True,
        trim_silence_flag: bool = True
    ) -> dict:
        """
        Processa todos os arquivos de áudio em um diretório.
        
        Args:
            input_dir: Diretório de entrada
            output_dir: Diretório de saída
            remove_noise_flag: Se True, remove ruído
            trim_silence_flag: Se True, remove silêncio
        
        Returns:
            Dicionário com estatísticas do processamento
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Busca arquivos de áudio
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(input_path.rglob(f'*{ext}')))
        
        print(f"Encontrados {len(audio_files)} arquivos de áudio")
        
        # Processa cada arquivo
        successful = 0
        failed = 0
        metadata = []
        
        for audio_file in tqdm(audio_files, desc="Processando áudios"):
            # Mantém estrutura de diretórios
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.wav')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            success = self.process_file(
                str(audio_file),
                str(output_file),
                remove_noise_flag=remove_noise_flag,
                trim_silence_flag=trim_silence_flag
            )
            
            if success:
                successful += 1
                metadata.append({
                    'original_file': str(audio_file),
                    'processed_file': str(output_file),
                    'status': 'success'
                })
            else:
                failed += 1
                metadata.append({
                    'original_file': str(audio_file),
                    'processed_file': str(output_file),
                    'status': 'failed'
                })
        
        # Salva metadata
        df = pd.DataFrame(metadata)
        df.to_csv(output_path / 'processing_metadata.csv', index=False)
        
        stats = {
            'total': len(audio_files),
            'successful': successful,
            'failed': failed
        }
        
        print(f"\nProcessamento concluído:")
        print(f"  Total: {stats['total']}")
        print(f"  Sucesso: {stats['successful']}")
        print(f"  Falhas: {stats['failed']}")
        
        return stats


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Pré-processa arquivos de áudio de tosse'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Diretório com áudios brutos'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Diretório para áudios processados'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração'
    )
    parser.add_argument(
        '--no-noise-reduction',
        action='store_true',
        help='Desabilita remoção de ruído'
    )
    parser.add_argument(
        '--no-trim-silence',
        action='store_true',
        help='Desabilita remoção de silêncio'
    )
    
    args = parser.parse_args()
    
    # Inicializa preprocessador
    preprocessor = AudioPreprocessor(config_path=args.config)
    
    # Processa diretório
    preprocessor.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        remove_noise_flag=not args.no_noise_reduction,
        trim_silence_flag=not args.no_trim_silence
    )


if __name__ == '__main__':
    main()

