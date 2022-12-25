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

	poetry run mypy segal

spell:
	poetry run codespell	

test:
	poetry run pytest tests
