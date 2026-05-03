.PHONY: test train train-cloud coverage evaluate config-show config-validate help-cli

test: clean
	@echo "Running tests..."
	PYTHONPATH=. pytest -x -v --tb=short -m "not slow"

all-tests: clean
	@echo "Running all tests..."
	PYTHONPATH=. pytest -v

local_train: clean
	@echo "Running local training..."
	PYTHONPATH=. python3 -m src.cli train --local

train: clean
	@echo "Running cloud training..."
	PYTHONPATH=. python3 -m src.cli train --cloud

coverage: clean
	@echo "Running tests with coverage..."
	@PYTHONPATH=. pytest --cov=src --cov-fail-under=95 --cov-report=term-missing -q && echo "\n✅ PASS: Test coverage is >= 95%" || (echo "\n❌ FAIL: Test coverage is < 95%" && exit 1)

eval: clean
	@echo "Running evaluation..."
	PYTHONPATH=. python3 -m src.cli evaluate

local_eval: clean
	@echo "Running local evaluation..."
	PYTHONPATH=. python3 -m src.cli evaluate --is_local

config-show: clean
	@echo "Showing configuration..."
	PYTHONPATH=. python3 -m src.cli config show

config-validate: clean
	@echo "Validating configuration..."
	PYTHONPATH=. python3 -m src.cli config validate

clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -f model.pt checkpoint_0.pt

check:
	@echo "Running linter..."
	ruff check src tests
	@echo "Checking code formatting..."
	ruff format --check src tests

fix:
	@echo "Formatting code..."
	ruff format src tests
	@echo "Running linter..."
	ruff check src tests --fix --unsafe-fixes

help:
	@echo "Showing help..."
	PYTHONPATH=. python3 -m src.cli --help