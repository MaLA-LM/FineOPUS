#!/bin/bash

nsamples=100
lang1code="abk_Cyrl"
lang2code="bos_Latn"
hfpath="MaLA-LM/FineOPUS-Deduplicated"
outputfolder="./data/samples"

python 0_create_sample_data.py \
    --hf_path ${hfpath} \
    --n_samples ${nsamples} \
    --lang1_code ${lang1code} \
    --lang2_code ${lang2code} \
    --output_folder ${outputfolder}

python 1_create_annotation_html.py 