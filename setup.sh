#!/bin/bash

# create env
rm -rf labyrintho_env
python -m venv labyrintho_env
source labyrintho_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Finished. Go Work!"