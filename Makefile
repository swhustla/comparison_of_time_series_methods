install:
	poetry shell
	pip install pip --upgrade
	pip install tensorflow-macos
	pip install git+https://github.com/stan-dev/pystan2.git@master
	pip install prophet==1.0      
	pip install GDAL
	pip install torch==1.13.1
	poetry install
lint:
	poetry run pylint -d duplicate-code ./**/*.py
run: install
	poetry run python ./main.py