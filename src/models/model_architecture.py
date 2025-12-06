"""
Definições de arquiteturas de modelos para classificação de tosse.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from typing import Tuple


def create_cnn_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Cria modelo CNN simples para classificação.
    
    Args:
        input_shape: Forma do input (altura, largura, canais)
        num_classes: Número de classes
        dropout_rate: Taxa de dropout
    
    Returns:
        Modelo Keras compilado
    """
    model = models.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.75),
        
        # Bloco 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Classificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate * 0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_Classifier')
    
    return model


def create_mobilenet_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3,
    trainable_base: bool = False
) -> keras.Model:
    """
    Cria modelo baseado em MobileNetV2 (otimizado para mobile).
    
    Args:
        input_shape: Forma do input (altura, largura, canais)
        num_classes: Número de classes
        dropout_rate: Taxa de dropout
        trainable_base: Se True, treina toda a base MobileNet
    
    Returns:
        Modelo Keras compilado
    """
    # Se input tem 1 canal, expande para 3 canais (requerido pelo MobileNet)
    inputs = layers.Input(shape=input_shape)
    
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
    else:
        x = inputs
    
    # Base MobileNetV2 pré-treinada
    base_model = MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base
    
    # Feature extraction
    x = base_model(x, training=False)
    
    # Classificador customizado
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='MobileNetV2_Classifier')
    
    return model


def create_lightweight_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Cria modelo leve otimizado para dispositivos móveis.
    
    Args:
        input_shape: Forma do input (altura, largura, canais)
        num_classes: Número de classes
        dropout_rate: Taxa de dropout
    
    Returns:
        Modelo Keras compilado
    """
    model = models.Sequential([
        # Bloco 1 - Depthwise Separable Conv
        layers.SeparableConv2D(32, (3, 3), activation='relu', 
                               input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloco 2
        layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Bloco 3
        layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Classificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name='Lightweight_Classifier')
    
    return model


def create_residual_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Cria modelo com blocos residuais.
    
    Args:
        input_shape: Forma do input (altura, largura, canais)
        num_classes: Número de classes
        dropout_rate: Taxa de dropout
    
    Returns:
        Modelo Keras compilado
    """
    inputs = layers.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Bloco Residual 1
    shortcut = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Bloco Residual 2
    shortcut = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Classificador
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='Residual_Classifier')
    
    return model


def get_model(
    architecture: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3,
    **kwargs
) -> keras.Model:
    """
    Factory function para criar modelo baseado no nome da arquitetura.
    
    Args:
        architecture: Nome da arquitetura ('cnn', 'mobilenet', 'lightweight', 'residual')
        input_shape: Forma do input
        num_classes: Número de classes
        dropout_rate: Taxa de dropout
        **kwargs: Argumentos adicionais específicos da arquitetura
    
    Returns:
        Modelo Keras
    """
    architectures = {
        'cnn': create_cnn_model,
        'mobilenet': create_mobilenet_model,
        'mobilenet_v2': create_mobilenet_model,
        'lightweight': create_lightweight_model,
        'residual': create_residual_model
    }
    
    if architecture.lower() not in architectures:
        raise ValueError(
            f"Arquitetura '{architecture}' não suportada. "
            f"Opções: {list(architectures.keys())}"
        )
    
    model_fn = architectures[architecture.lower()]
    model = model_fn(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        **kwargs
    )
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    metrics: list = None
) -> keras.Model:
    """
    Compila modelo com otimizador e métricas.
    
    Args:
        model: Modelo Keras
        learning_rate: Taxa de aprendizado
        metrics: Lista de métricas (default: accuracy, precision, recall, auc)
    
    Returns:
        Modelo compilado
    """
    if metrics is None:
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    return model

