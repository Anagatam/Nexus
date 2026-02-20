.PHONY: install test format lint clean

# Default python command
PYTHON = python3

# Install dependencies (standard + enterprise)
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

install-enterprise:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[enterprise,dev]"

# Execute the core test pipeline locally
test:
	$(PYTHON) -m pytest tests/
	
# Run the structural demo as an integration test
verify:
	$(PYTHON) -m run_nexus_demo

# Format codebase using Black
format:
	black nexus/
	black tests/

# Execute PEP8 linting
lint:
	flake8 nexus/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 nexus/ --count --max-complexity=10 --max-line-length=127 --statistics

# Clean the workspace
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
