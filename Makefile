install:
	poetry install
lint:
	poetry run pylint -d duplicate-code ./**/*.py
run: install
	poetry run ./main.py