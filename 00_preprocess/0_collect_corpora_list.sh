#!/bin/bash
# Collect OPUS corpora list in Moses format, process it to get corpus info, then clean up.

echo "Querying OPUS for available corpora in Moses format..."
opus_get -l -p moses > all_corpora_moses.txt

echo "Processing the corpus list using the Python script..."
python3 get_corpora_list.py

echo "Cleaning up temporary files..."
rm all_corpora_moses.txt

echo "Cloning OPUS repository for obtaining language mapping..."
git clone https://github.com/Helsinki-NLP/OPUS.git

echo "Done."
