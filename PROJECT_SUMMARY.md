# Resumo do Projeto: Análise de Tosse para Detecção de Pneumonia e Bronquite

## 🎯 Objetivo

Desenvolver uma aplicação móvel capaz de analisar o som da tosse e auxiliar na identificação de pneumonia e bronquite através de Machine Learning, processando dados localmente no dispositivo para garantir privacidade.

## 📊 Visão Geral Técnica

### Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      APLICAÇÃO MÓVEL                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Interface do Usuário (Flutter)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Gravador de Áudio (16kHz, mono)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Pré-processamento Local                  │  │
│  │  • Normalização    • Remoção de ruído                │  │
│  │  • Mel Spectrogram • Feature extraction              │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Modelo TFLite (CNN - MobileNetV2)             │  │
│  │     Quantizado INT8 para performance móvel           │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Predição (3 classes)                     │  │
│  │     Normal | Bronquite | Pneumonia                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🗂️ Estrutura do Projeto

```
cough-analysis-app/
├── data/                      # Dados e modelos
│   ├── raw/                   # Áudios brutos
│   ├── processed/             # Áudios processados
│   └── models/                # Modelos treinados
│
├── src/                       # Código-fonte Python
│   ├── preprocessing/         # Pipeline de dados
│   │   ├── prepare_data.py    # Pré-processamento
│   │   ├── extract_features.py # Extração de features
│   │   └── data_augmentation.py # Data augmentation
│   │
│   ├── models/                # Modelos ML
│   │   ├── model_architecture.py # Definição de modelos
│   │   ├── train_model.py     # Script de treinamento
│   │   └── convert_to_tflite.py # Conversão para mobile
│   │
│   └── utils/                 # Utilitários
│       ├── audio_utils.py     # Manipulação de áudio
│       └── visualization.py   # Visualizações
│
├── mobile/cough_detector/     # Aplicação Flutter
│   ├── lib/
│   │   ├── screens/           # Telas
│   │   ├── services/          # Serviços (audio, ML)
│   │   └── widgets/           # Widgets customizados
│   └── assets/                # Recursos (modelo TFLite)
│
├── notebooks/                 # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb
│
├── docs/                      # Documentação
│   ├── GETTING_STARTED.md     # Guia de início
│   └── DATASETS.md            # Guia de datasets
│
├── config.yaml                # Configuração do pipeline
├── requirements.txt           # Dependências Python
├── Makefile                   # Automação de tarefas
└── README.md                  # Documentação principal
```

## 🧠 Pipeline de Machine Learning

### 1. Pré-processamento de Áudio

**Entrada:** Arquivos de áudio brutos (.wav, .mp3, etc.)

**Operações:**
- Reamostragem para 16 kHz
- Conversão para mono
- Normalização de amplitude
- Remoção de ruído (spectral gating)
- Remoção de silêncio
- Padding/truncamento para duração fixa (5s)

**Saída:** Arquivos WAV normalizados (16kHz, 16-bit, mono)

### 2. Extração de Características

**Features Extraídas:**

1. **MFCC** (Mel Frequency Cepstral Coefficients)
   - 40 coeficientes
   - Captura timbre e textura do som
   - Inclui deltas (1ª e 2ª derivadas)

2. **Mel Spectrogram** (128x128)
   - Representação visual tempo-frequência
   - Escala mel (mais próxima da percepção humana)
   - Usado como input para CNN

3. **Características Espectrais**
   - Spectral centroid (centro de massa do espectro)
   - Spectral rolloff (frequência de corte)
   - Spectral bandwidth (largura do espectro)
   - Spectral contrast (diferença entre picos e vales)

4. **Zero Crossing Rate**
   - Taxa de cruzamento por zero
   - Indica irregularidades no sinal

5. **Características Temporais**
   - RMS energy (energia do sinal)
   - Tempo estimado
   - Duração de eventos

**Formato de Saída:**
- Feature vector: ~200 features numéricas
- Mel spectrogram: array 128x128 normalizado [0,1]

### 3. Modelo de Deep Learning

**Arquitetura: MobileNetV2 Adaptado**

```
Input (128x128x1 - Mel Spectrogram)
    ↓
Conv2D 1x1 (expande para 3 canais)
    ↓
MobileNetV2 Base (pré-treinada ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU)
    ↓
BatchNormalization + Dropout(0.3)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.15)
    ↓
Dense(3, Softmax)
    ↓
Output: [P(Normal), P(Bronquite), P(Pneumonia)]
```

**Características:**
- **Parâmetros:** ~2.5M (após quantização)
- **Tamanho:** ~3MB (TFLite INT8)
- **Inferência:** ~50-100ms em smartphone médio
- **Acurácia esperada:** 75-85% (depende do dataset)

**Treinamento:**
- Otimizador: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Métricas: Accuracy, Precision, Recall, AUC
- Early Stopping (paciência: 15 épocas)
- ReduceLROnPlateau (paciência: 7 épocas)
- Class weights para balanceamento

### 4. Otimização para Mobile

**Conversão para TensorFlow Lite:**
- Quantização INT8 (reduz 4x o tamanho)
- Representative dataset para calibração
- Otimizações de operadores
- Teste de benchmark de performance

**Performance esperada:**
- Tamanho do modelo: 2-4 MB
- Latência: 50-150 ms
- Uso de memória: < 50 MB
- Compatibilidade: Android 7+ / iOS 12+

## 📱 Aplicação Móvel (Flutter)

### Funcionalidades

1. **Gravação de Áudio**
   - Duração: 2-5 segundos
   - Formato: WAV 16kHz mono
   - Visualização em tempo real (waveform)
   - Permissões gerenciadas automaticamente

2. **Processamento Local**
   - Extração de mel spectrogram
   - Normalização
   - Inferência com TFLite

3. **Resultados**
   - Classe predita com confiança
   - Probabilidades detalhadas para cada classe
   - Recomendações baseadas no resultado
   - Interface visual intuitiva

4. **Privacidade**
   - **Todos os dados processados localmente**
   - Nenhum áudio enviado para servidores
   - Opção de excluir gravações

### Tecnologias Utilizadas

**Flutter/Dart:**
- `record` - Gravação de áudio
- `tflite_flutter` - Inferência TFLite
- `provider` - State management
- `permission_handler` - Permissões

## 📊 Datasets Recomendados

1. **COUGHVID** - 25K+ amostras de tosse
2. **ICBHI** - 920 sons respiratórios com diagnósticos
3. **Coswara** - Sons de tosse, respiração e fala
4. **VIRUFY** - Tosse para detecção de COVID-19

## 🔬 Métricas de Avaliação

- **Accuracy** - Acurácia geral
- **Precision** - Precisão por classe
- **Recall** - Sensibilidade por classe
- **F1-Score** - Média harmônica
- **AUC-ROC** - Área sob a curva
- **Confusion Matrix** - Matriz de confusão

## ⚠️ Limitações e Considerações

### Limitações Técnicas

1. **Qualidade do áudio**
   - Ruído ambiental afeta performance
   - Microfones de baixa qualidade degradam resultados

2. **Variabilidade**
   - Tosses variam muito entre indivíduos
   - Idade, sexo, comorbidades influenciam

3. **Dataset**
   - Datasets públicos são limitados
   - Desbalanceamento de classes comum
   - Viés geográfico/demográfico

### Considerações Clínicas

1. **NÃO é diagnóstico médico**
   - Ferramenta auxiliar apenas
   - Requer validação profissional

2. **Especificidade vs. Sensibilidade**
   - Trade-off entre falsos positivos e negativos
   - Ajustável via threshold

3. **Validação clínica necessária**
   - Estudos prospectivos
   - Aprovação regulatória

## 🚀 Próximos Passos

### Curto Prazo
- [ ] Coletar/integrar mais datasets
- [ ] Aumentar data augmentation
- [ ] Testar arquiteturas alternativas
- [ ] Melhorar UI/UX do app

### Médio Prazo
- [ ] Implementar modelo ensemble
- [ ] Adicionar detecção de qualidade de áudio
- [ ] Sistema de feedback do usuário
- [ ] Backend opcional para analytics

### Longo Prazo
- [ ] Validação clínica em hospitais
- [ ] Publicação científica
- [ ] Certificação médica (FDA, ANVISA)
- [ ] Multi-idioma e acessibilidade

## 📈 Métricas de Sucesso

**Técnicas:**
- Acurácia > 80%
- Latência < 200ms
- Tamanho do app < 50MB

**Clínicas:**
- Sensibilidade > 85% (detectar casos verdadeiros)
- Especificidade > 75% (evitar falsos alarmes)
- Validação por médicos especialistas

**Produto:**
- Taxa de retenção > 40%
- Avaliação na loja > 4.5/5
- Impacto social mensurável

## 👥 Equipe e Contribuições

Este é um projeto open-source. Contribuições são bem-vindas!

**Áreas para contribuir:**
- Machine Learning
- Mobile Development
- UI/UX Design
- Documentação
- Coleta de dados
- Validação clínica

## 📄 Licença

MIT License - Software livre para uso, modificação e distribuição.

## 🙏 Agradecimentos

Agradeço à comunidade open-source pelos datasets, ferramentas e conhecimento compartilhado que tornaram este projeto possível.

---

**Data de criação:** Dezembro 2025  
**Versão:** 1.0.0  
**Status:** Protótipo funcional

