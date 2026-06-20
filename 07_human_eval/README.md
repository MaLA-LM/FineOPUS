# A static human evaluation interface

## 1. Prerequisites

The only dependancy is the `datasets` library, which is used to load data from Hugging Face.

```bash
pip install datasets

```

## 2. Streaming and sampling data

The script `0_create_sample_data.py` streams data in both `lang1code-lang2code` and `lang2code-lang1code` directions from the data path provided, by default, `MaLA-LM/FineOPUS-Filtered-Stage4`. The script then samples from the stream using reservoir sampling. The sampled data is then saved in the `annotation_samples` folder. The script can be run as follows:

```bash

python 0_create_sample_data.py \
    --hf_path ${hf_path} \
    --n_samples ${n_samples} \
    --lang1_code ${lang1code} \
    --lang2_code ${lang2code} \
    --output_folder ${annotation_data_path}

```

---
## 3. Creating static annotation HTML files

For all sample data under `annotation_samples`, the script `1_create_annotation_html.py` can be used to create static annotation HTML files. The script automatically reads all sample data and creates 1) an index page linking to all language pairs' interface; 2) a CSS file for styling; and 3) individual annotation pages for each language pair. These are static pages and do not require a backend server. The script can be run as follows:

```bash
python 1_create_annotation_html.py \
    --input_path ${annotation_data_path} \
    --output_path ${annotation_html_path}
```

The advantage of pre-computing static HTML files is that 1) the data pool to sample from is usually very large, so separating the true sampling process from the annotation task allows for fast interface rendering; 2) it does not require a backend server, so it can be easily deployed or even sent to an annotator as html files for them to open in their browser.

There are two known drawbacks: 1) since there is no backend server, the annotation results are not returned but are saved in the browser's local storage. 2) Currently, there is no automated way to generate interfaces for a new language pair as an annotator onboarded. As a mitigation, the process can be semi-automated by running a bash script that 1) loops through all languages of interest while calling `0_create_sample_data.py`; then 2) creates HTMLs pages using `1_create_annotation_html.py`.

---
## 4. Annotation justification

The nature of the data being annotated are filtered parallel corpus, which is generally for training MT and LLMs, rather than for direct human consumption. Therefore, instead of running a fine-grained error analysis which requires linguistic expertise, we decide to evaluate three aspects: 1) the languages are correct with respect to the language codes; 2) sentences are fluent and natural; 3) the sentences are translations of each other. This annotation work simply requires bilingual or profient speakers of the two languages involved.
