# Aplicação de Análise de Tosse - Detecção de Pneumonia e Bronquite

## 📋 Descrição

Aplicação móvel capaz de analisar o som da tosse e auxiliar na identificação de pneumonia e bronquite através de análise de áudio com Machine Learning.

## 🎯 Objetivos

- Detectar padrões sonoros associados à pneumonia e bronquite
- Fornecer análise em tempo real no dispositivo móvel
- Garantir privacidade dos dados de saúde do usuário

## 🏗️ Arquitetura do Sistema

```
Aplicativo Móvel
     |
 [Audio Recorder]
     |
[Pré-processamento]
 - Normalização
 - Remoção de ruído
 - Geração de espectrograma
     |
[Modelo TFLite no dispositivo]
     |
[Predição]
     |
Interface de resultado
```

## 📁 Estrutura do Projeto

```
cough-analysis-app/
├── data/
│   ├── raw/              # Dados de áudio brutos
│   ├── processed/        # Dados processados
│   └── models/           # Modelos treinados
├── notebooks/            # Jupyter notebooks para experimentação
├── src/
│   ├── preprocessing/    # Scripts de pré-processamento
│   ├── models/           # Definição e treinamento de modelos
│   └── utils/            # Utilitários
├── mobile/
│   └── cough_detector/   # Aplicação Flutter
└── docs/                 # Documentação adicional
```

## 🔧 Instalação

### Requisitos

- Python 3.8+
- TensorFlow 2.x
- Flutter 3.x (para aplicação móvel)

### Setup do Ambiente Python

```bash
cd cough-analysis-app
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Setup da Aplicação Móvel

```bash
cd mobile/cough_detector
flutter pub get
```

## 🗃️ Datasets Recomendados

1. **COUGHVID Dataset** - EPFL (https://coughvid.epfl.ch/)
2. **ICBHI Respiratory Sound Database** - Sons respiratórios incluindo bronquite e pneumonia
3. **Coswara Dataset** - Sons de tosse, respiração e fala
4. **VIRUFY Cough Dataset** - Foco em doenças respiratórias
5. **ICSR** - International Cough Sound Repository

## 🚀 Como Usar

### 1. Preparar os Dados

```bash
python src/preprocessing/prepare_data.py --input data/raw --output data/processed
```

### 2. Extrair Características

```bash
python src/preprocessing/extract_features.py --input data/processed --output data/features
```

### 3. Treinar o Modelo

```bash
python src/models/train_model.py --data data/features --output data/models
```

### 4. Converter para TFLite

```bash
python src/models/convert_to_tflite.py --model data/models/best_model.h5 --output mobile/cough_detector/assets
```

### 5. Executar a Aplicação Móvel

```bash
cd mobile/cough_detector
flutter run
```

## 📊 Características Extraídas

- **MFCC** (Mel Frequency Cepstral Coefficients) - Captura timbre e forma do espectro
- **Log-Mel Spectrogram** - Representação visual para CNNs
- **Zero Crossing Rate** - Identifica intensidade e irregularidades
- **Chroma Features** - Análise de tonalidade
- **Características temporais** - Duração, explosividade, ritmo

## 🧠 Modelo

- **Arquitetura**: CNN baseada em MobileNetV2 (otimizada para mobile)
- **Input**: Log-Mel Spectrogram (128x128)
- **Output**: 3 classes (Normal, Bronquite, Pneumonia)
- **Otimização**: TensorFlow Lite com quantização INT8

## ⚠️ Aviso Importante

**Esta aplicação NÃO substitui diagnóstico médico profissional.** É uma ferramenta auxiliar que deve ser usada em conjunto com avaliação médica adequada.

## 📝 Licença

MIT License

## 👥 Contribuições

Contribuições são bem-vindas! Por favor, consulte as diretrizes de contribuição.

## 📧 Contato

Para questões e suporte, abra uma issue no repositório.

