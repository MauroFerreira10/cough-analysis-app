"""
Script para converter modelo TensorFlow para TensorFlow Lite.
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))


class TFLiteConverter:
    """Classe para converter modelos para TensorFlow Lite."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o conversor.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def convert_model(
        self,
        model_path: str,
        output_path: str,
        quantization: str = 'int8',
        representative_dataset: np.ndarray = None
    ) -> str:
        """
        Converte modelo para TFLite.
        
        Args:
            model_path: Caminho para modelo .h5
            output_path: Caminho para salvar modelo .tflite
            quantization: Tipo de quantização ('int8', 'float16', 'none')
            representative_dataset: Dataset representativo para quantização
        
        Returns:
            Caminho do modelo convertido
        """
        print(f"Carregando modelo de {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Cria conversor
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configura otimizações
        if quantization == 'int8':
            print("Aplicando quantização INT8...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_dataset is not None:
                def representative_dataset_gen():
                    for i in range(min(100, len(representative_dataset))):
                        yield [representative_dataset[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            
        elif quantization == 'float16':
            print("Aplicando quantização FLOAT16...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        elif quantization == 'none':
            print("Sem quantização...")
        
        else:
            raise ValueError(f"Tipo de quantização inválido: {quantization}")
        
        # Converte
        print("Convertendo modelo...")
        tflite_model = converter.convert()
        
        # Salva
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Mostra informações
        original_size = os.path.getsize(model_path)
        converted_size = os.path.getsize(output_path)
        compression_ratio = (1 - converted_size / original_size) * 100
        
        print(f"\nConversão concluída!")
        print(f"  Modelo original: {original_size / 1024:.2f} KB")
        print(f"  Modelo TFLite: {converted_size / 1024:.2f} KB")
        print(f"  Compressão: {compression_ratio:.1f}%")
        print(f"  Salvo em: {output_path}")
        
        return str(output_path)
    
    def test_tflite_model(
        self,
        tflite_path: str,
        test_input: np.ndarray
    ) -> np.ndarray:
        """
        Testa modelo TFLite com input de exemplo.
        
        Args:
            tflite_path: Caminho para modelo .tflite
            test_input: Input de teste
        
        Returns:
            Output do modelo
        """
        print(f"\nTestando modelo TFLite...")
        
        # Carrega modelo
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Obtém detalhes de input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output dtype: {output_details[0]['dtype']}")
        
        # Prepara input
        if input_details[0]['dtype'] == np.uint8:
            # Quantiza input para uint8
            input_scale, input_zero_point = input_details[0]['quantization']
            test_input = test_input / input_scale + input_zero_point
            test_input = test_input.astype(np.uint8)
        else:
            test_input = test_input.astype(np.float32)
        
        # Roda inferência
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantiza output se necessário
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        print(f"Output: {output}")
        print(f"Predicted class: {np.argmax(output)}")
        
        return output
    
    def benchmark_model(
        self,
        tflite_path: str,
        test_input: np.ndarray,
        num_runs: int = 100
    ) -> dict:
        """
        Faz benchmark de performance do modelo TFLite.
        
        Args:
            tflite_path: Caminho para modelo .tflite
            test_input: Input de teste
            num_runs: Número de execuções para benchmark
        
        Returns:
            Dicionário com estatísticas de performance
        """
        import time
        
        print(f"\nExecutando benchmark ({num_runs} runs)...")
        
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepara input
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            test_input = test_input / input_scale + input_zero_point
            test_input = test_input.astype(np.uint8)
        else:
            test_input = test_input.astype(np.float32)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        stats = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }
        
        print(f"\nResultados do Benchmark:")
        print(f"  Média: {stats['mean_ms']:.2f} ms")
        print(f"  Desvio padrão: {stats['std_ms']:.2f} ms")
        print(f"  Mínimo: {stats['min_ms']:.2f} ms")
        print(f"  Máximo: {stats['max_ms']:.2f} ms")
        print(f"  Mediana: {stats['median_ms']:.2f} ms")
        
        return stats


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Converte modelo TensorFlow para TensorFlow Lite'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Caminho para modelo .h5'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Caminho para salvar modelo .tflite'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['int8', 'float16', 'none'],
        default='int8',
        help='Tipo de quantização'
    )
    parser.add_argument(
        '--representative-data',
        type=str,
        help='Caminho para dados representativos (.pkl) para quantização'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Testa modelo após conversão'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Faz benchmark do modelo'
    )
    
    args = parser.parse_args()
    
    # Inicializa conversor
    converter = TFLiteConverter(config_path=args.config)
    
    # Carrega dados representativos se fornecidos
    representative_dataset = None
    if args.representative_data:
        print(f"Carregando dados representativos de {args.representative_data}...")
        with open(args.representative_data, 'rb') as f:
            data = pickle.load(f)
            if 'mel_spectrograms' in data:
                mel_specs = data['mel_spectrograms'][:100]
                # Processa para shape correto
                representative_dataset = np.array([
                    tf.image.resize(spec[:, :, np.newaxis], (128, 128)).numpy()
                    for spec in mel_specs
                ])
    
    # Converte modelo
    tflite_path = converter.convert_model(
        model_path=args.model,
        output_path=args.output,
        quantization=args.quantization,
        representative_dataset=representative_dataset
    )
    
    # Testa se solicitado
    if args.test or args.benchmark:
        # Cria input de teste
        test_input = np.random.randn(1, 128, 128, 1).astype(np.float32)
        
        if args.test:
            converter.test_tflite_model(tflite_path, test_input)
        
        if args.benchmark:
            converter.benchmark_model(tflite_path, test_input)


if __name__ == '__main__':
    main()

