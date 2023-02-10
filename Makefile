all: install clean tests
clean:
	git clean -Xdf
install:
	pip install .
tests:
	pip install .[testing]
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	pytest
upload-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
upload: dist
	twine upload dist/*
format:
	autoflake .
	isort .
	black .
	yapf -i -r .
dist:
	python -m build
docs:
	pip install .[docs]
	sphinx-build sphinx docs -b html
	touch docs/.nojekyll
