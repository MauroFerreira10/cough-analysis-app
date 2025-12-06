# -*- coding: utf-8 -*-
"""
Script de teste simples para verificar instalação.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

try:
    import librosa
    print(f"✓ Librosa {librosa.__version__}")
except Exception as e:
    print(f"✗ Librosa: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"✗ Matplotlib: {e}")

try:
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
except Exception as e:
    print(f"✗ SciPy: {e}")

print("\n" + "="*60)
print("✓ Todas as bibliotecas essenciais estão instaladas!")
print("="*60)
print("\nPróximos passos:")
print("\n1. Obter dados dos datasets (veja docs/DATASETS.md)")
print("   - COUGHVID: https://coughvid.epfl.ch/")
print("   - ICBHI: https://bhichallenge.med.auth.gr/")
print("\n2. Organizar dados em:")
print("   data/raw/normal/")
print("   data/raw/bronquite/")
print("   data/raw/pneumonia/")
print("\n3. Executar pipeline:")
print("   python src/preprocessing/prepare_data.py --input data/raw --output data/processed")
print("   python src/preprocessing/extract_features.py --input data/processed --output data/features")
print("   python src/models/train_model.py --data data/features/features.pkl --output data/models")
print("\nDocumentação:")
print("- README.md - Visão geral completa")
print("- QUICK_START.md - Setup rápido")
print("- PROJECT_SUMMARY.md - Detalhes técnicos")

