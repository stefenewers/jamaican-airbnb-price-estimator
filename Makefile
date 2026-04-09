.PHONY: install train test lint docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python -m ml.train

test:
	pytest tests/ -v --tb=short --cov=ml --cov=api --cov-report=term-missing

lint:
	python -m py_compile config.py run.py ml/features.py ml/pipeline.py ml/train.py api/app.py api/routes/predict.py api/routes/health.py
	@echo "Syntax check passed"

docker-build:
	docker build -t airbnb-price-predictor:latest .

docker-run:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage; \
	echo "Cleaned."
