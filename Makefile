.PHONY: install train api test lint fmt docker-build docker-run

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python src/train.py

api:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

lint:
	flake8 .
	black --check .

fmt:
	black .

docker-build:
	docker build -t iris-ml-api:latest .

docker-run:
	docker run -p 8000:8000 iris-ml-api:latest
