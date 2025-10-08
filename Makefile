all: install clean tests
clean:
	git clean -Xdf
install:
	pip install .
tests:
	uv sync --group testing
	ruff check .
	@echo "Running parallel tests with pytest-xdist..."
	uv run pytest -m parallel -n auto
	@echo "Running sequential tests..."
	uv run pytest -m "not parallel"
upload-test: dist
	uv publish --publish-url https://test.pypi.org/legacy/
upload: dist
	uv publish
format:
	uv run ruff format .
lint:
	uv run ruff check .
dist:
	uv build
docs:
	uv sync --group docs
	uv run sphinx-build sphinx docs -b html
	touch docs/.nojekyll
