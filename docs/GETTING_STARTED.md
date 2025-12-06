# Guia de Início Rápido

## 📋 Visão Geral

Este guia irá ajudá-lo a configurar e executar o projeto de análise de tosse do zero.

## 🔧 Pré-requisitos

### Software Necessário

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **Flutter 3.0+** (para aplicação móvel)
   ```bash
   flutter --version
   ```

3. **Git**
   ```bash
   git --version
   ```

### Hardware Recomendado

- **Para treinamento**: GPU com suporte CUDA (opcional, mas recomendado)
- **Para aplicação móvel**: Dispositivo Android/iOS ou emulador

## 📦 Instalação

### 1. Clone o Repositório

```bash
git clone <repository-url>
cd cough-analysis-app
```

### 2. Configure o Ambiente Python

```bash
# Crie ambiente virtual
python -m venv venv

# Ative o ambiente
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt
```

### 3. Verifique a Instalação

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import librosa; print(librosa.__version__)"
```

## 📊 Obtendo os Dados

### Datasets Recomendados

1. **COUGHVID Dataset**
   - Website: https://coughvid.epfl.ch/
   - Baixe e extraia para `data/raw/coughvid/`

2. **ICBHI Respiratory Sound Database**
   - Website: https://bhichallenge.med.auth.gr/
   - Baixe e extraia para `data/raw/icbhi/`

### Estrutura de Diretórios

Organize os dados assim:

```
data/
├── raw/
│   ├── normal/
│   │   └── *.wav
│   ├── bronquite/
│   │   └── *.wav
│   └── pneumonia/
│       └── *.wav
├── processed/
└── models/
```

## 🚀 Pipeline de Desenvolvimento

### Passo 1: Pré-processamento

```bash
python src/preprocessing/prepare_data.py \
  --input data/raw \
  --output data/processed
```

**O que faz:**
- Normaliza taxa de amostragem (16kHz)
- Remove ruído
- Remove silêncio
- Padroniza duração

### Passo 2: Extração de Características

```bash
python src/preprocessing/extract_features.py \
  --input data/processed \
  --output data/features \
  --labels "normal:normal,bronquite:bronquite,pneumonia:pneumonia"
```

**O que extrai:**
- MFCCs (40 coeficientes)
- Mel Spectrograms
- Chroma features
- Zero Crossing Rate
- Características espectrais

### Passo 3: Treinamento do Modelo

```bash
python src/models/train_model.py \
  --data data/features/features.pkl \
  --output data/models \
  --architecture mobilenet_v2
```

**Arquiteturas disponíveis:**
- `cnn` - CNN simples
- `mobilenet_v2` - MobileNetV2 (recomendado para mobile)
- `lightweight` - Modelo leve customizado
- `residual` - Rede residual

**Monitoramento:**
```bash
tensorboard --logdir data/models/logs
```

### Passo 4: Conversão para TFLite

```bash
python src/models/convert_to_tflite.py \
  --model data/models/best_model.h5 \
  --output mobile/cough_detector/assets/model.tflite \
  --quantization int8 \
  --representative-data data/features/features.pkl \
  --test \
  --benchmark
```

**Tipos de quantização:**
- `int8` - Menor tamanho, mais rápido (recomendado)
- `float16` - Meio termo
- `none` - Sem quantização

## 📱 Executando a Aplicação Móvel

### Setup

```bash
cd mobile/cough_detector
flutter pub get
```

### Executar em Dispositivo/Emulador

```bash
# Listar dispositivos
flutter devices

# Executar
flutter run
```

### Build para Produção

**Android:**
```bash
flutter build apk --release
# APK estará em: build/app/outputs/flutter-apk/app-release.apk
```

**iOS:**
```bash
flutter build ios --release
```

## 🔬 Experimentação com Notebooks

### Iniciar Jupyter

```bash
jupyter notebook notebooks/
```

### Notebooks Disponíveis

1. `01_exploratory_analysis.ipynb` - Análise exploratória de dados
2. (Adicione mais conforme necessário)

## 🧪 Testando o Sistema

### Teste Rápido com Áudio de Exemplo

```python
from src.preprocessing.extract_features import FeatureExtractor
from tensorflow import keras
import numpy as np

# Carregar modelo
model = keras.models.load_model('data/models/best_model.h5')

# Extrair features
extractor = FeatureExtractor()
result = extractor.process_file('caminho/para/tosse.wav')

# Predição
prediction = model.predict(result['mel_spectrogram'][np.newaxis, ...])
print(f"Classe predita: {np.argmax(prediction)}")
print(f"Probabilidades: {prediction}")
```

## 📈 Melhorando o Modelo

### Data Augmentation

```python
from src.preprocessing.data_augmentation import AudioAugmenter

augmenter = AudioAugmenter(sample_rate=16000)
augmented = augmenter.random_augment(audio, n_augmentations=5)
```

### Ajuste Fino (Fine-tuning)

Para melhorar o modelo com novos dados:

```bash
python src/models/train_model.py \
  --data data/features/new_features.pkl \
  --output data/models/finetuned \
  --architecture mobilenet_v2 \
  --load-weights data/models/best_model.h5
```

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"

Reduza o batch size no `config.yaml`:

```yaml
training:
  batch_size: 16  # Era 32
```

### Erro: "No audio files found"

Verifique estrutura de diretórios e extensões de arquivo (.wav, .mp3).

### Erro: Modelo TFLite não carrega no Flutter

1. Verifique se `model.tflite` está em `assets/`
2. Confirme que está listado no `pubspec.yaml`
3. Execute: `flutter clean && flutter pub get`

### Erro: Permissão de microfone negada

**Android:** Verifique `AndroidManifest.xml`  
**iOS:** Verifique `Info.plist`

## 📚 Recursos Adicionais

### Documentação

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flutter Documentation](https://flutter.dev/)
- [Librosa Documentation](https://librosa.org/)

### Papers Relevantes

1. "Deep Learning for Audio Signal Processing" - IEEE
2. "Cough Sound Analysis for COVID-19 Detection" - Nature
3. "Mobile Health Applications using ML" - ACM

## 💡 Dicas

1. **Comece pequeno**: Use subset dos dados para testes rápidos
2. **Monitore overfitting**: Use early stopping e dropout
3. **Valide com dados reais**: Grave suas próprias amostras para testar
4. **Documente experimentos**: Use MLflow ou WandB
5. **Versione os modelos**: Mantenha histórico de experimentos

## 🤝 Contribuindo

Leia `CONTRIBUTING.md` para diretrizes de contribuição.

## 📧 Suporte

- Issues: GitHub Issues
- Discussões: GitHub Discussions
- Email: [seu-email]

## 📄 Licença

MIT License - veja `LICENSE` para detalhes.

