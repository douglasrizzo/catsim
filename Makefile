PACKNAME = catsim
all: install clean tests
clean:
	git clean -Xdf
install:
	pip install .
tests:
	pip install .[testing]

ifneq '$USER' 'travis'
	nosetests -s --cov-config .coveragerc --with-coverage --cover-package=catsim --processes=$(nproc)
else
	nosetests -s --cov-config .coveragerc --with-coverage --cover-package=catsim
endif
upload-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
upload:
	twine upload dist/*
format: clean
	-yapf -i -r .
