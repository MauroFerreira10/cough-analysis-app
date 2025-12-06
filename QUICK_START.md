# Guia de Início Rápido ⚡

## Setup em 5 minutos

### 1. Clone e Configure (2 min)

```bash
# Clone o repositório
git clone <repository-url>
cd cough-analysis-app

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt
```

### 2. Teste com Dados Sintéticos (1 min)

```bash
# Execute demonstração
python demo.py --synthetic --output demos/
```

Isso irá:
- ✅ Criar áudio sintético de tosse
- ✅ Pré-processar o áudio
- ✅ Extrair características
- ✅ Gerar visualizações

### 3. Explore com Jupyter (2 min)

```bash
# Inicie Jupyter
jupyter notebook notebooks/

# Abra: 01_exploratory_analysis.ipynb
```

## Pipeline Completo (com seus dados)

### Passo 1: Organize os Dados

```
data/raw/
├── normal/
│   └── *.wav
├── bronquite/
│   └── *.wav
└── pneumonia/
    └── *.wav
```

### Passo 2: Execute Pipeline

```bash
# Makefile torna tudo mais fácil!

# Pré-processa áudios
make preprocess

# Extrai características
make extract-features

# Treina modelo
make train

# Converte para TFLite
make convert
```

### Passo 3: App Móvel

```bash
# Setup Flutter
cd mobile/cough_detector
flutter pub get

# Execute
flutter run
```

## Comandos Úteis

```bash
# Ver todos os comandos disponíveis
make help

# Limpar arquivos temporários
make clean

# Executar testes
make test

# Visualizar treinamento
make tensorboard
```

## Arquivos Importantes

| Arquivo | Descrição |
|---------|-----------|
| `config.yaml` | Configurações do pipeline |
| `demo.py` | Script de demonstração |
| `Makefile` | Automação de tarefas |
| `README.md` | Documentação completa |

## Próximos Passos

1. 📚 Leia `docs/GETTING_STARTED.md` para detalhes
2. 📊 Veja `docs/DATASETS.md` para obter dados
3. 🔬 Explore `notebooks/` para experimentação
4. 📱 Customize `mobile/cough_detector/` para seu caso

## Problemas Comuns

**Erro: ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Erro: Permission denied**
```bash
chmod +x demo.py
```

**Erro: CUDA out of memory**
```yaml
# Em config.yaml
training:
  batch_size: 16  # Reduza de 32
```

## Ajuda

- 📖 Documentação: Veja `docs/`
- 💬 Issues: GitHub Issues
- 📧 Email: [seu-email]

## ⚠️ Lembre-se

**Este app NÃO substitui diagnóstico médico!**

Sempre consulte um profissional de saúde.

---

Pronto! Agora você está pronto para começar. 🚀

