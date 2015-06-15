PACKNAME = catsim
all: install clean
clean:
	sudo rm -rf $(PACKNAME).egg-info dist build
	find -name __pycache__ -exec rm -rf {} \;
install:
	sudo python3 setup.py install
tests:
	nosetests
