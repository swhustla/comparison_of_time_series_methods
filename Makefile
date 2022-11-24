install:
	poetry shell
	pip install pip --upgrade
	pip install pystan==2.19.1.1
	pip install torch==1.12.1
	poetry install
lint:
	poetry run pylint -d duplicate-code ./**/*.py
run: install
	poetry run python ./main.py