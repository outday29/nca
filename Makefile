.PHONY: format lint test

format:
	black .
	ruff --select I --fix .

lint:
	black . --check
	ruff .

test:
	pytest ./tests -s