main:
	python3 main.py
init:
	pip install -r requirements.txt
git:
	git config --local user.name "Alireza Ghafouri"
	git config --local user.email "alirezaghafouri98@gmail.com"
rm_cache:
	find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} \;
black:
	black .
pipreqs:
	pipreqs --force .