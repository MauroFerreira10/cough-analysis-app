#!/bin/bash
# Script de inicialização do projeto

echo "======================================"
echo "  PROJETO: Análise de Tosse com ML"
echo "======================================"
echo ""

# Verifica Python
if command -v python3 &> /dev/null; then
    echo "✓ Python3 encontrado: $(python3 --version)"
else
    echo "✗ Python3 não encontrado"
    echo "  Instale em: https://python.org"
    exit 1
fi

# Verifica Flutter
if command -v flutter &> /dev/null; then
    echo "✓ Flutter encontrado: $(flutter --version | head -n 1)"
else
    echo "⚠ Flutter não encontrado (opcional para mobile)"
fi

echo ""
echo "======================================"
echo "  ARQUIVOS DO PROJETO"
echo "======================================"
echo ""
echo "📂 Estrutura:"
echo "  - 11 módulos Python (src/)"
echo "  - 7 arquivos Flutter (mobile/)"
echo "  - 16 arquivos documentação"
echo ""

echo "📚 Documentação Principal:"
echo "  1. README.md - Visão geral"
echo "  2. QUICK_START.md - Setup rápido"
echo "  3. EXECUTIVE_SUMMARY.md - Resumo executivo"
echo "  4. PROJECT_SUMMARY.md - Detalhes técnicos"
echo "  5. STATUS.md - Status atual"
echo ""

echo "======================================"
echo "  PRÓXIMOS PASSOS"
echo "======================================"
echo ""
echo "1. Leia a documentação:"
echo "   cat README.md"
echo ""
echo "2. Configure ambiente Python:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Obtenha dados (veja docs/DATASETS.md):"
echo "   - COUGHVID: https://coughvid.epfl.ch/"
echo "   - ICBHI: https://bhichallenge.med.auth.gr/"
echo ""
echo "4. Execute pipeline:"
echo "   python src/preprocessing/prepare_data.py --input data/raw --output data/processed"
echo ""
echo "======================================"
echo "  SUPORTE"
echo "======================================"
echo ""
echo "Problemas? Veja:"
echo "  - STATUS.md - Status e troubleshooting"
echo "  - EXECUTIVE_SUMMARY.md - Guia completo"
echo "  - docs/GETTING_STARTED.md - Tutorial"
echo ""
echo "✅ Projeto pronto para desenvolvimento!"
echo ""

