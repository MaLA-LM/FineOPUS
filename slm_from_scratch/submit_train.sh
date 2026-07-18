#!/bin/bash

set -euo pipefail

###############################################################################
# Training jobs
#
# Add one entry per training job, using exactly this field order:
#   train-script|WANDB_PROJECT|WANDB_EXP_NAME|_DATA_MIX
# The Slurm job name is set automatically to WANDB_EXP_NAME.
#
# _DATA_MIX uses Megatron's alternating "weight path" format. For example:
#   0.7 /path/to/data_a 0.3 /path/to/data_b
###############################################################################
RUN_CONFIGS=(
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_multilingual_FineOPUS|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/multilingual_mix/_bin/FineOPUS/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_multilingual_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/multilingual_mix/_bin/MaLA_Bi/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_multilingual_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/multilingual_mix/_bin/NLLB/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ara_Arab_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-ara_Arab/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ara_Arab_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-ara_Arab/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ara_Arab_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-ara_Arab/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ara_Arab_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-ara_Arab/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ara_Arab_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-ara_Arab/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ara_Arab_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-ara_Arab/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-bul_Cyrl_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-bul_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-bul_Cyrl_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-bul_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-bul_Cyrl_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-bul_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-bul_Cyrl_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-bul_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-bul_Cyrl_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-bul_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-bul_Cyrl_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-bul_Cyrl/combined_text_document'    

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-deu_Latn_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-deu_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-deu_Latn_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-deu_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-deu_Latn_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-deu_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-deu_Latn_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-deu_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-deu_Latn_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-deu_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-deu_Latn_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-deu_Latn/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ell_Grek_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-ell_Grek/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ell_Grek_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-ell_Grek/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ell_Grek_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-ell_Grek/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ell_Grek_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-ell_Grek/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ell_Grek_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-ell_Grek/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ell_Grek_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-ell_Grek/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-fra_Latn_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-fra_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-fra_Latn_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-fra_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-fra_Latn_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-fra_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-fra_Latn_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-fra_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-fra_Latn_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-fra_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-fra_Latn_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-fra_Latn/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ita_Latn_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-ita_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ita_Latn_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-ita_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ita_Latn_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-ita_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ita_Latn_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-ita_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ita_Latn_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-ita_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ita_Latn_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-ita_Latn/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-por_Latn_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-por_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-por_Latn_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-por_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-por_Latn_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-por_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-por_Latn_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-por_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-por_Latn_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-por_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-por_Latn_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-por_Latn/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ron_Latn_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-ron_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ron_Latn_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-ron_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ron_Latn_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-ron_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ron_Latn_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-ron_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ron_Latn_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-ron_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-ron_Latn_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-ron_Latn/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-rus_Cyrl_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-rus_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-rus_Cyrl_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-rus_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-rus_Cyrl_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-rus_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-rus_Cyrl_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-rus_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-rus_Cyrl_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-rus_Cyrl/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-rus_Cyrl_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-rus_Cyrl/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-spa_Latn_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-spa_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-spa_Latn_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-spa_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-spa_Latn_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-spa_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-spa_Latn_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-spa_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-spa_Latn_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-spa_Latn/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-spa_Latn_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-spa_Latn/combined_text_document'

    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-zho_Hans_FineOPUS-Stage1|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage1/eng_Latn-zho_Hans/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-zho_Hans_FineOPUS-Stage2|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage2/eng_Latn-zho_Hans/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-zho_Hans_FineOPUS-Stage3|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage3/eng_Latn-zho_Hans/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-zho_Hans_FineOPUS-Stage4|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/FineOPUS-Filtered-Stage4/eng_Latn-zho_Hans/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-zho_Hans_MaLA_Bi|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/MaLA-Bi/eng_Latn-zho_Hans/combined_text_document'
    # 'train_0_4B.sh|FineOPUS_Models|0.4B_Pretrain_eng_Latn-zho_Hans_NLLB|1.0 /scratch/project_462001427/FineOPUS/slm_from_scratch/data/combined/bilingual_mix/_bin/NLLB/eng_Latn-zho_Hans/combined_text_document'

    # 0.9B example (remove the leading # and edit the dataset before use):
    # 'train_0_9B.sh|FineOPUS_Models|0.9B_Pretrain_multilingual_FineOPUS|1.0 /path/to/0.9B/dataset_prefix'
)

usage() {
    cat <<'EOF'
Usage: ./submit_train.sh

Edit RUN_CONFIGS near the top of this script to add training jobs. Running the
script once submits every entry as an independent Slurm job.

Entry format:
  train-script|WANDB_PROJECT|WANDB_EXP_NAME|_DATA_MIX

The Slurm job name is set automatically to WANDB_EXP_NAME.

Supported training scripts:
  train_0_4B.sh
  train_0_9B.sh
EOF
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: this script does not accept command-line arguments" >&2
            usage >&2
            exit 2
            ;;
    esac
fi

[[ ${#RUN_CONFIGS[@]} -gt 0 ]] || {
    echo "Error: RUN_CONFIGS is empty" >&2
    exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Validate every configuration before submitting any jobs. This avoids a typo
# in a later entry leaving the batch only partially submitted.
for config in "${RUN_CONFIGS[@]}"; do
    IFS='|' read -r train_script_name wandb_project wandb_exp_name data_mix extra <<< "$config"

    if [[ -n "${extra:-}" || -z "$train_script_name" || -z "$wandb_project" || -z "$wandb_exp_name" || -z "$data_mix" ]]; then
        echo "Error: invalid RUN_CONFIGS entry:" >&2
        echo "  $config" >&2
        echo "Expected: train-script|WANDB_PROJECT|WANDB_EXP_NAME|_DATA_MIX" >&2
        exit 2
    fi

    case "$train_script_name" in
        train_0_4B.sh|train_0_9B.sh) ;;
        *)
            echo "Error: unsupported training script: $train_script_name" >&2
            exit 2
            ;;
    esac

    if [[ ! -f "$SCRIPT_DIR/$train_script_name" ]]; then
        echo "Error: training script not found: $SCRIPT_DIR/$train_script_name" >&2
        exit 2
    fi

    if [[ "$wandb_exp_name" == *'/'* ]]; then
        echo "Error: WANDB_EXP_NAME must not contain '/': $wandb_exp_name" >&2
        exit 2
    fi
done

echo "Submitting ${#RUN_CONFIGS[@]} training job(s)..."

for config in "${RUN_CONFIGS[@]}"; do
    IFS='|' read -r train_script_name wandb_project wandb_exp_name data_mix <<< "$config"
    job_name="$wandb_exp_name"
    train_script="$SCRIPT_DIR/$train_script_name"
    mkdir -p "$SCRIPT_DIR/logs/train/$job_name"

    echo
    echo "Submitting $job_name"
    echo "  TRAIN_SCRIPT=$train_script_name"
    echo "  _DATA_MIX=$data_mix"
    echo "  WANDB_PROJECT=$wandb_project"
    echo "  WANDB_EXP_NAME=$wandb_exp_name"

    _DATA_MIX="$data_mix" \
    WANDB_PROJECT="$wandb_project" \
    WANDB_EXP_NAME="$wandb_exp_name" \
    sbatch --chdir="$SCRIPT_DIR" --export=ALL --job-name="$job_name" "$train_script"
done
