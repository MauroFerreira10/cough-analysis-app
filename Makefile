# Makefile para automação de tarefas comuns

.PHONY: help install clean test lint format train convert run-app

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       - Instala dependências Python"
	@echo "  make clean         - Limpa arquivos temporários"
	@echo "  make test          - Executa testes"
	@echo "  make lint          - Verifica código com flake8"
	@echo "  make format        - Formata código com black"
	@echo "  make train         - Treina modelo (exemplo)"
	@echo "  make convert       - Converte modelo para TFLite"
	@echo "  make run-app       - Executa aplicação Flutter"

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/

test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=100

format:
	black src/ --line-length=100

# Pipeline de treinamento
preprocess:
	python src/preprocessing/prepare_data.py \
		--input data/raw \
		--output data/processed

extract-features:
	python src/preprocessing/extract_features.py \
		--input data/processed \
		--output data/features \
		--labels "normal:normal,bronquite:bronquite,pneumonia:pneumonia"

train:
	python src/models/train_model.py \
		--data data/features/features.pkl \
		--output data/models \
		--architecture mobilenet_v2

convert:
	python src/models/convert_to_tflite.py \
		--model data/models/best_model.h5 \
		--output mobile/cough_detector/assets/model.tflite \
		--quantization int8 \
		--test \
		--benchmark

# Aplicação móvel
flutter-setup:
	cd mobile/cough_detector && flutter pub get

run-app:
	cd mobile/cough_detector && flutter run

build-apk:
	cd mobile/cough_detector && flutter build apk --release

# Jupyter
jupyter:
	jupyter notebook notebooks/

# TensorBoard
tensorboard:
	tensorboard --logdir data/models/logs

