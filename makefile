.PHONY: setup format lint test run-app

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pre-commit install

format:
	black . && isort .

lint:
	flake8 .

test:
	pytest -q || true

run-app:
	streamlit run app/app.py