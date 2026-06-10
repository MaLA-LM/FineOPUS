# A static human evaluation interface

## 🚀 Quick Start

### 1. Prerequisites

The only dependancy is the `datasets` library, which is used to load data from Hugging Face.

```bash
pip install datasets

```

### 2. Streaming and sampling data

Given a `lang_Script` code pair, `0_create_sample_data.py` streams data from the particular folder containing this code pair in the data path provided, by default, `MaLA-LM/FineOPUS-Deduplicated`. The script then pseudo-samples from the stream by using a buffer of a certain size; then it is randomly shuffled. The sampled data is then saved in the `data/samples` folder as: `sample_{lang1}_{Script1}_{lang2}_{Script2}_{sample_size}.jsonl`. The script can be manually run as follows:

```bash
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

```

---
### 3. Creating static annotation HTML files

For all sample data under `data/samples`, the script `1_create_annotation_html.py` can be used to create static annotation HTML files. The script automatically reads all sample data and creates 1) an index page linking to all language pairs' annotation interface; 2) a CSS file for styling; and 3) individual annotation pages for each language pair, which can be opened in a browser. The script can be manually run as follows:

```bash
python 1_create_annotation_html.py 
```

The created HTML files will be saved in the `data/annotation_html` folder. A working prototype is currently hosted at https://pinzhenchen.github.io/prototype/ (may be removed at any time). The advantage of pre-computing static HTML files is that 1) it allows for a very fast and responsive annotation interface and 2) it does not require a backend server. 

There are two known drawbacks: 1) since there is no backend server, the annotation results are not returned but are saved in the browser's local storage. Annotators can export their annotations as a JSONL file, which can be loaded back to the browser to continue their progress or sent back to us once done. 2) Currently, there is no automated way to generate interfaces for a new language pair. However, the process can be easily automated by running a bash script that calls `0_create_sample_data.py` by looping through all languages of interest and then create all HTMLs using `1_create_annotation_html.py`.

A potential TODO is that we can use a cloud storage (e.g., AWS S3) to store the annotation results, which can be submitted by the annotators through the interface. This way, we can collect annotations from multiple annotators automatically.

---
### 4. Annotation work

The nature of the data being annotated are filtered parallel corpus, which is generally for training MT and LLMs, rather than for direct human consumption. Therefore, instead of running a fine-grained error analysis (`v1`) which requires linguistic expertise, this version `v2` evaluates three aspects: 1) the languages are correct wrt the language codes; 2) sentences are fluent and natural; 3) the sentences are translations of each other. Therefore, this annotation simply requires bilingual or profient speakers of the two languages.