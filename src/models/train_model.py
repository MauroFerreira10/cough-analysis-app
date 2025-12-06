"""
Script para treinar modelo de classificação de tosse.
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_architecture import get_model, compile_model


class CoughClassifierTrainer:
    """Classe para treinar classificador de tosse."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o trainer.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
    
    def load_data(self, features_path: str) -> tuple:
        """
        Carrega dados de características.
        
        Args:
            features_path: Caminho para arquivo pickle com features
        
        Returns:
            Tupla (mel_spectrograms, labels)
        """
        print(f"Carregando dados de {features_path}...")
        
        with open(features_path, 'rb') as f:
            data = pickle.load(f)
        
        mel_specs = data['mel_spectrograms']
        labels = data['labels']
        
        print(f"Carregados {len(mel_specs)} exemplos")
        print(f"Classes encontradas: {set(labels)}")
        
        return mel_specs, labels
    
    def preprocess_spectrograms(
        self,
        mel_specs: list,
        target_shape: tuple = (128, 128)
    ) -> np.ndarray:
        """
        Pré-processa espectrogramas para tamanho uniforme.
        
        Args:
            mel_specs: Lista de espectrogramas Mel
            target_shape: Forma alvo (altura, largura)
        
        Returns:
            Array numpy com espectrogramas processados
        """
        print("Pré-processando espectrogramas...")
        
        processed = []
        for spec in mel_specs:
            # Resize para target_shape
            spec_resized = tf.image.resize(
                spec[:, :, np.newaxis],
                target_shape
            ).numpy()
            processed.append(spec_resized)
        
        return np.array(processed)
    
    def prepare_dataset(
        self,
        features_path: str,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> tuple:
        """
        Prepara datasets de treino, validação e teste.
        
        Args:
            features_path: Caminho para arquivo de features
            test_size: Proporção para teste
            val_size: Proporção para validação (do conjunto de treino)
            random_state: Seed para reprodutibilidade
        
        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Carrega dados
        mel_specs, labels = self.load_data(features_path)
        
        # Remove exemplos sem label
        valid_indices = [i for i, label in enumerate(labels) if label is not None]
        mel_specs = [mel_specs[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        # Processa espectrogramas
        input_shape = self.config['model']['input_shape']
        X = self.preprocess_spectrograms(mel_specs, target_shape=input_shape[:2])
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        print(f"\nClasses codificadas:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")
        
        # Split treino/teste
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Split treino/validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_val
        )
        
        print(f"\nDistribuição dos dados:")
        print(f"  Treino: {len(X_train)} exemplos")
        print(f"  Validação: {len(X_val)} exemplos")
        print(f"  Teste: {len(X_test)} exemplos")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_callbacks(self, output_dir: str) -> list:
        """
        Cria callbacks para treinamento.
        
        Args:
            output_dir: Diretório para salvar checkpoints e logs
        
        Returns:
            Lista de callbacks
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Salva melhor modelo
            keras.callbacks.ModelCheckpoint(
                filepath=str(output_path / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduz learning rate
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['training']['reduce_lr_factor'],
                patience=self.config['training']['reduce_lr_patience'],
                min_lr=self.config['training']['min_lr'],
                verbose=1
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=str(output_path / 'logs'),
                histogram_freq=1
            ),
            
            # CSV Logger
            keras.callbacks.CSVLogger(
                filename=str(output_path / 'training_log.csv')
            )
        ]
        
        return callbacks
    
    def train(
        self,
        features_path: str,
        output_dir: str,
        architecture: str = None
    ) -> dict:
        """
        Treina o modelo.
        
        Args:
            features_path: Caminho para arquivo de features
            output_dir: Diretório para salvar modelo e logs
            architecture: Arquitetura do modelo (se None, usa do config)
        
        Returns:
            Dicionário com histórico de treinamento
        """
        # Prepara dados
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_dataset(
            features_path
        )
        
        # Calcula class weights para lidar com desbalanceamento
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"\nClass weights: {class_weight_dict}")
        
        # Cria modelo
        if architecture is None:
            architecture = self.config['model']['architecture']
        
        print(f"\nCriando modelo: {architecture}")
        self.model = get_model(
            architecture=architecture,
            input_shape=tuple(self.config['model']['input_shape']),
            num_classes=self.config['model']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        # Compila modelo
        self.model = compile_model(
            self.model,
            learning_rate=self.config['model']['learning_rate']
        )
        
        # Mostra resumo
        self.model.summary()
        
        # Cria callbacks
        callbacks = self.create_callbacks(output_dir)
        
        # Treina
        print("\nIniciando treinamento...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Avalia no conjunto de teste
        print("\nAvaliando no conjunto de teste...")
        test_results = self.model.evaluate(X_test, y_test, verbose=1)
        
        print("\nResultados no teste:")
        for metric_name, value in zip(self.model.metrics_names, test_results):
            print(f"  {metric_name}: {value:.4f}")
        
        # Salva modelo final
        output_path = Path(output_dir)
        self.model.save(str(output_path / 'final_model.h5'))
        
        # Salva label encoder
        with open(output_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Salva configuração
        with open(output_path / 'model_config.json', 'w') as f:
            json.dump({
                'architecture': architecture,
                'input_shape': self.config['model']['input_shape'],
                'num_classes': self.config['model']['num_classes'],
                'classes': self.label_encoder.classes_.tolist(),
                'test_results': {
                    name: float(value)
                    for name, value in zip(self.model.metrics_names, test_results)
                }
            }, f, indent=2)
        
        print(f"\nModelo salvo em {output_dir}")
        
        return self.history.history


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Treina modelo de classificação de tosse'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Caminho para arquivo de features (.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Diretório para salvar modelo'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        choices=['cnn', 'mobilenet', 'lightweight', 'residual'],
        help='Arquitetura do modelo (sobrescreve config)'
    )
    
    args = parser.parse_args()
    
    # Inicializa trainer
    trainer = CoughClassifierTrainer(config_path=args.config)
    
    # Treina modelo
    history = trainer.train(
        features_path=args.data,
        output_dir=args.output,
        architecture=args.architecture
    )
    
    print("\nTreinamento concluído!")


if __name__ == '__main__':
    main()

