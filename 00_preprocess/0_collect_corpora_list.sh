#!/bin/bash

echo "Processing the corpus list using the Python script..."
python3 get_corpora_list.py

echo "Cloning OPUS repository for obtaining language mapping..."
git clone https://github.com/Helsinki-NLP/OPUS.git

echo "Done."
