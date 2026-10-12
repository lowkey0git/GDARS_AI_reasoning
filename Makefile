# ============================================================
# GDARS v2 Makefile
# Linux/Mac: make <command>
# Windows PowerShell: see equivalent commands in README.md
# ============================================================

.PHONY: help setup run test test-cov docker-up docker-down clean tf-init tf-plan tf-apply

help:
	@echo ""
	@echo "GDARS v2 commands:"
	@echo "  make setup       — install deps + create dirs + copy .env"
	@echo "  make run         — start API server (requires .env)"
	@echo "  make test        — run full test suite"
	@echo "  make test-cov    — run tests with coverage report"
	@echo "  make docker-up   — start API + Redis in Docker"
	@echo "  make docker-down — stop containers"
	@echo "  make clean       — remove cache and artifacts"
	@echo "  make tf-init     — terraform init"
	@echo "  make tf-plan     — terraform plan"
	@echo "  make tf-apply    — terraform apply"
	@echo ""

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	mkdir -p data/chroma data/faiss_index logs
	@[ -f .env ] || (cp .env.example .env && echo "Created .env — add your API keys")

run:
	python main.py

test:
	PYTHONPATH=. pytest tests/ -v --tb=short -x

test-cov:
	PYTHONPATH=. pytest tests/ -v \
		--cov=agents --cov=memory --cov=tools --cov=evaluation --cov=utils \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

docker-up:
	docker compose up --build -d
	@echo "API running at http://localhost:8000"

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage coverage.xml htmlcov

tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan -var-file=terraform.tfvars

tf-apply:
	cd terraform && terraform apply -var-file=terraform.tfvars
