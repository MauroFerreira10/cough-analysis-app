# Assets

Coloque o arquivo `model.tflite` aqui após treinar e converter o modelo.

## Como obter o modelo:

1. Treine o modelo:
```bash
cd ../../
python src/models/train_model.py --data data/features/features.pkl --output data/models
```

2. Converta para TFLite:
```bash
python src/models/convert_to_tflite.py \
  --model data/models/best_model.h5 \
  --output mobile/cough_detector/assets/model.tflite \
  --quantization int8
```

3. O arquivo `model.tflite` será criado automaticamente neste diretório.

