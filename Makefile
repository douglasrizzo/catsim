PACKNAME = catsim
all: install clean tests
clean:
	rm -rf $(PACKNAME).egg-info dist build
	find -name *.pyc -delete
	find -name __pycache__ -delete
install:
	pip3 install .
tests:
	pip3 install .[testing]

ifneq '$USER' 'travis'
	nosetests -s --cov-config .coveragerc --with-coverage --cover-package=catsim --processes=$(nproc)
else
	nosetests -s --cov-config .coveragerc --with-coverage --cover-package=catsim
endif
upload-test:
	python setup.py register -r pypitest && python setup.py sdist upload -r pypitest
upload:
	python setup.py register -r pypi && python setup.py sdist upload -r pypi
format: clean
	-yapf -i -r .
