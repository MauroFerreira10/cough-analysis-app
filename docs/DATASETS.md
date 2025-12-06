# Guia de Datasets para Análise de Tosse

## 📊 Datasets Públicos Disponíveis

### 1. COUGHVID Dataset

**Descrição:** Um dos maiores datasets de tosse com sintomas respiratórios, coletado durante a pandemia de COVID-19.

**Conteúdo:**
- Mais de 25.000 gravações de tosse
- Labels: tosse saudável, COVID-19, sintomas respiratórios
- Metadata: idade, sexo, local, sintomas

**Como obter:**
1. Acesse: https://coughvid.epfl.ch/
2. Preencha formulário de requisição
3. Download via Zenodo

**Citação:**
```
Orlandic, L., Teijeiro, T., & Atienza, D. (2021). 
The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms.
Scientific Data, 8(1), 1-10.
```

**Uso neste projeto:**
```bash
# Estrutura esperada
data/raw/coughvid/
├── audio/
│   └── *.wav
└── metadata.csv
```

---

### 2. ICBHI Respiratory Sound Database

**Descrição:** Database clássica de sons respiratórios incluindo várias patologias.

**Conteúdo:**
- 920 gravações de sons respiratórios
- Anotações de ciclos respiratórios
- Doenças: COPD, bronquite, pneumonia, asma, etc.
- 126 pacientes

**Como obter:**
1. Acesse: https://bhichallenge.med.auth.gr/
2. Registre-se para acesso
3. Download do dataset

**Citação:**
```
Rocha, B. M., et al. (2019).
An open access database for the evaluation of respiratory sound classification algorithms.
Physiological Measurement, 40(3), 035001.
```

**Uso neste projeto:**
```bash
# Estrutura esperada
data/raw/icbhi/
├── audio_and_txt_files/
│   ├── *.wav
│   └── *.txt
└── patient_diagnosis.csv
```

---

### 3. Coswara Dataset

**Descrição:** Dataset de sons respiratórios, tosse, respiração e fala para detecção de doenças.

**Conteúdo:**
- Sons de tosse (superficial e profunda)
- Sons de respiração
- Contagem de números
- Metadata de saúde

**Como obter:**
1. Acesse: https://github.com/iiscleap/Coswara-Data
2. Clone repositório
3. Download via script fornecido

**Citação:**
```
Sharma, N., et al. (2020).
Coswara - A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis.
arXiv preprint arXiv:2005.10548.
```

---

### 4. VIRUFY Cough Dataset

**Descrição:** Dataset focado em detecção de COVID-19 através de tosse.

**Conteúdo:**
- Gravações de tosse de múltiplos países
- Labels de teste COVID
- Metadata demográfica

**Como obter:**
1. Acesse: https://virufy.org/
2. Contate a equipe para acesso ao dataset
3. Siga termos de uso

---

### 5. ESC-50 (Environmental Sound Classification)

**Descrição:** Dataset genérico de sons ambientais, útil para pré-treinamento.

**Conteúdo:**
- 2.000 gravações de sons ambientais
- 50 classes diferentes
- Inclui sons de tosse e respiração

**Como obter:**
1. Acesse: https://github.com/karolpiczak/ESC-50
2. Download direto ou via Kaggle

---

### 6. FSD50K (Freesound Dataset 50K)

**Descrição:** Large-scale dataset de sons gerais, útil para transfer learning.

**Conteúdo:**
- 51.197 clipes de áudio
- 200 classes
- Inclui sons humanos e respiratórios

**Como obter:**
1. Acesse: https://zenodo.org/record/4060432
2. Download via Zenodo

---

## 🔄 Pré-processamento de Datasets

### Script Unificado

```python
# src/preprocessing/prepare_datasets.py

import os
import pandas as pd
from pathlib import Path

def prepare_coughvid(input_dir, output_dir):
    """Prepara COUGHVID dataset"""
    # Lê metadata
    metadata = pd.read_csv(f"{input_dir}/metadata.csv")
    
    # Filtra por qualidade
    metadata = metadata[metadata['cough_detected'] > 0.8]
    
    # Mapeia labels
    label_map = {
        'healthy': 'normal',
        'COVID-19': 'pneumonia',  # Simplificação
        'symptomatic': 'bronquite'  # Simplificação
    }
    
    # Processa arquivos
    for idx, row in metadata.iterrows():
        src = f"{input_dir}/audio/{row['uuid']}.wav"
        dst_label = label_map.get(row['status'], 'unknown')
        dst = f"{output_dir}/{dst_label}/{row['uuid']}.wav"
        # Copiar e processar arquivo
        
def prepare_icbhi(input_dir, output_dir):
    """Prepara ICBHI dataset"""
    # Lê diagnósticos
    diagnosis = pd.read_csv(f"{input_dir}/patient_diagnosis.csv")
    
    # Mapeia doenças para labels
    label_map = {
        'Healthy': 'normal',
        'Bronchitis': 'bronquite',
        'Pneumonia': 'pneumonia',
        'COPD': 'bronquite',
        'Asthma': 'bronquite'
    }
    
    # Processa arquivos
    # ...
```

---

## 📋 Formato de Metadata

### Template CSV

```csv
file_path,label,duration,sample_rate,quality_score,source,patient_id
data/processed/normal/001.wav,normal,2.5,16000,0.95,coughvid,P001
data/processed/bronquite/002.wav,bronquite,3.2,16000,0.88,icbhi,P045
data/processed/pneumonia/003.wav,pneumonia,2.8,16000,0.92,coswara,P102
```

---

## 🔍 Análise de Qualidade

### Checklist de Validação

```python
def validate_dataset(data_dir):
    """Valida qualidade do dataset"""
    
    checks = {
        'sample_rate': [],
        'duration': [],
        'snr': [],  # Signal-to-noise ratio
        'clipping': []
    }
    
    for audio_file in Path(data_dir).rglob('*.wav'):
        audio, sr = librosa.load(audio_file)
        
        # Verifica taxa de amostragem
        checks['sample_rate'].append(sr == 16000)
        
        # Verifica duração
        duration = len(audio) / sr
        checks['duration'].append(1.0 <= duration <= 5.0)
        
        # Verifica SNR
        snr = calculate_snr(audio)
        checks['snr'].append(snr > 10)  # dB
        
        # Verifica clipping
        clipped = np.sum(np.abs(audio) > 0.99) / len(audio)
        checks['clipping'].append(clipped < 0.01)
    
    return checks
```

---

## 🌐 Coletando Seus Próprios Dados

### Protocolo de Coleta

1. **Consentimento:**
   - Obtenha consentimento informado
   - Siga regulações de privacidade (GDPR, LGPD)

2. **Ambiente:**
   - Sala silenciosa (< 40 dB ruído de fundo)
   - Distância: 15-30cm do microfone
   - Sem reverberação excessiva

3. **Instruções ao Participante:**
   - Tossir naturalmente
   - 3-5 tosses por sessão
   - Diferentes intensidades

4. **Configuração de Gravação:**
   ```yaml
   sample_rate: 16000  # Hz
   bit_depth: 16
   channels: 1  # Mono
   format: WAV
   duration: 5  # segundos
   ```

5. **Metadata a Coletar:**
   - Idade
   - Sexo
   - Diagnóstico médico (se aplicável)
   - Sintomas atuais
   - Data da gravação
   - Dispositivo usado

### Script de Gravação

```python
import sounddevice as sd
import soundfile as sf
from datetime import datetime

def record_cough(duration=5, output_dir='data/raw/custom'):
    """Grava tosse do microfone"""
    
    sr = 16000
    print(f"Gravando por {duration} segundos...")
    
    # Grava áudio
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    
    # Salva arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/cough_{timestamp}.wav"
    sf.write(filename, audio, sr)
    
    print(f"Salvo em: {filename}")
    return filename
```

---

## 📊 Balanceamento de Classes

### Estratégias

1. **Oversampling:**
   ```python
   from imblearn.over_sampling import SMOTE
   
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   ```

2. **Data Augmentation:**
   - Time stretching
   - Pitch shifting
   - Adicionar ruído
   - Time shifting

3. **Class Weights:**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight(
       'balanced',
       classes=np.unique(y_train),
       y=y_train
   )
   ```

---

## ⚖️ Considerações Éticas

### Checklist

- [ ] Consentimento informado obtido
- [ ] Dados anonimizados
- [ ] Armazenamento seguro
- [ ] Conformidade com regulações (GDPR, HIPAA, LGPD)
- [ ] Transparência no uso dos dados
- [ ] Direito de exclusão implementado
- [ ] Auditoria de viés no dataset

---

## 📚 Referências e Recursos

1. **AudioSet** - https://research.google.com/audioset/
2. **Zenodo** - https://zenodo.org/ (vários datasets de áudio)
3. **Kaggle** - https://www.kaggle.com/datasets (competições de áudio)
4. **PhysioNet** - https://physionet.org/ (sinais médicos)

---

## 🔄 Atualização de Datasets

Este documento será atualizado conforme novos datasets se tornem disponíveis.

**Última atualização:** Dezembro 2025

