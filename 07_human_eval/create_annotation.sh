#!/bin/bash

nsamples=100

# change the language codes
lang1code="abk_Cyrl"
lang2code="bos_Latn"

hfpath="MaLA-LM/FineOPUS-Filtered-Stage4"

outputdatafolder="./annotation_samples"
outputhtmlfolder="./annotation"

# create annotation samples by streaming the FineOPUS-Filtered-Stage4 
# dataset (in both forward and reverse directions) and reservoir sampling.
# this script is generic
python 0_create_sample_data.py \
    --hf_path ${hfpath} \
    --n_samples ${nsamples} \
    --lang1_code ${lang1code} \
    --lang2_code ${lang2code} \
    --output_folder ${outputdatafolder}

# cross-check the FineOPUS-Original dataset (original)
# against the FineOPUS-Filtered-Stage4 dataset (final)
# to find unique items in FineOPUS-Original.
# this script is heavily hard-coded
python 0_create_sample_data_by_crosschecking.py \
    --lang1_code ${lang1code} \

# create an HTML file for human evaluation of the annotation samples.
python 1_create_annotation_html.py \
    --input_path ${outputdatafolder} \
    --output_path ${outputhtmlfolder}

