init:
	poetry install

lint:
	poetry run black segal	
	poetry run black examples
	poetry run black tests

	poetry run isort segal
	poetry run isort examples
	poetry run isort tests

	poetry run flake8 segal
	poetry run flake8 examples
	poetry run flake8 tests 

test:
	poetry run pytest tests
