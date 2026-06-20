#!/bin/bash

nsamples=100
lang1code="abk_Cyrl"
lang2code="bos_Latn"
hfpath="MaLA-LM/FineOPUS-Filtered-Stage4"
outputdatafolder="./annotation_samples"
outputhtmlfolder="./annotation_htmls"

python 0_create_sample_data.py \
    --hf_path ${hfpath} \
    --n_samples ${nsamples} \
    --lang1_code ${lang1code} \
    --lang2_code ${lang2code} \
    --output_folder ${outputdatafolder}

python 1_create_annotation_html.py \
    --input_path ${outputdatafolder} \
    --output_path ${outputhtmlfolder}