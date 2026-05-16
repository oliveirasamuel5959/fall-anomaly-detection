run-train:
	uv run src/training/train.py

run-inference:
	uv run src/app/inference.py

run-collector:
	uv run src/app/main.py