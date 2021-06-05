all: install clean tests
clean:
	git clean -Xdf
install:
	pip install .
tests:
	pip install .[testing]
	nosetests -s --cov-config .coveragerc --with-coverage --cover-package=catsim
upload-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
upload: dist
	twine upload dist/*
format:
	-yapf -i -r catsim
dist:
	python -m build
docs:
	pip install .[docs]
	sphinx-build sphinx docs
