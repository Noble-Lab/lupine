help:
	@echo "lint - check code style with flake8"
	@echo "test - run tests only"
	@echo "coverage - run tests and check code coverage"
	@echo "conda_install (recommended) - Install requirements "

test:
	py.test

coverage:
	coverage run --source python_boilerplate --omit="*/test*" --module py.test
	coverage report --show-missing

lint:
	flake8 --exclude docs python_boilerplate

conda_install:
	conda install --file conda_requirements.txt
	pip install -r requirements.txt
	pip install .

install:
	pip install -r requirements.txt
	pip install .
