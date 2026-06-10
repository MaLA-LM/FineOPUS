#!/bin/bash

src="deu_Latn" # source language code
tgt="eng_Latn" # target language code

buffer=100000 # buffer size for pseudo-sampling. larger = more randomness but more memory usage

python 0_create_sample_data.py \
    --hf_path "MaLA-LM/FineOPUS-Deduplicated" \
    --n_samples 100 \
    --src_code ${src} \
    --tgt_code ${tgt} \
    --output_folder "./data/samples" \
    --buffer_size ${buffer} 

python 1_create_annotation_html.py 