#!/bin/bash
#SBATCH --job-name=download_fineweb2
#SBATCH --output=../logs/download_fineweb2/%x_%j.out
#SBATCH --error=../logs/download_fineweb2/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462001087
# Download selected FineWeb-2 language subsets from the Hugging Face Hub.
# Repo: HuggingFaceFW/fineweb-2  (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
#
# Usage:
#   ./download_fineweb2.sh                 # download train (+test) for all languages below
#   INCLUDE_REMOVED=1 ./download_fineweb2.sh   # also fetch the large "removed" (dedup) shards
#   LANGS="fra_Latn spa_Latn" ./download_fineweb2.sh   # override the language list
#
# Run this on a node with internet access (e.g. a LUMI login node).

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
source ../.venv/bin/activate
REPO_ID="HuggingFaceFW/fineweb-2"
DEST_ROOT="/scratch/project_462001069/fineweb-2"

# Pick whichever HF CLI is available (prefer the modern `hf`).
if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' found on PATH." >&2
    echo "       Activate the venv or 'pip install huggingface_hub[cli]'." >&2
    exit 1
fi

# Cache/tmp on scratch so we never fill up $HOME.
export HF_HOME="${HF_HOME:-/scratch/project_462001069/.cache/huggingface}"
# hf_transfer is deprecated in huggingface_hub >=1.x; use Xet high-performance mode.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
# Number of parallel download workers.
WORKERS="${WORKERS:-8}"

# Set INCLUDE_REMOVED=1 to also download the (very large) dedup "removed" shards.
INCLUDE_REMOVED="${INCLUDE_REMOVED:-0}"

# Languages to download (override by exporting LANGS="lang1 lang2 ...").
LANGS="${LANGS:-"
dzo_Tibt bem_Latn ssw_Latn pus_Arab mlg_Latn uzb_Latn bod_Tibt wol_Latn
yor_Latn nya_Latn nso_Latn tir_Ethi lav_Latn hau_Latn aze_Latn fao_Latn
fij_Latn tsn_Latn ewe_Latn zul_Latn nep_Deva mri_Latn smo_Latn est_Latn
sqi_Latn sna_Latn ibo_Latn bak_Cyrl swa_Latn isl_Latn lao_Laoo tat_Cyrl
kin_Latn xho_Latn mya_Mymr snd_Arab uig_Arab ltz_Latn tgk_Cyrl kat_Geor
glg_Latn som_Latn khm_Khmr mkd_Cyrl kor_Hang kir_Cyrl pan_Guru msa_Latn
sin_Sinh mlt_Latn gle_Latn fas_Arab hrv_Latn cym_Latn amh_Ethi hye_Armn
cat_Latn ukr_Cyrl urd_Arab srp_Latn heb_Hebr lit_Latn afr_Latn kaz_Cyrl
slv_Latn hin_Deva mar_Deva fin_Latn bel_Cyrl slk_Latn hun_Latn tur_Latn
fil_Latn zho_Hans dan_Latn kan_Knda guj_Gujr bul_Cyrl vie_Latn tam_Taml
ces_Latn bos_Latn ben_Beng tel_Telu mal_Mlym ron_Latn ell_Grek swe_Latn
ind_Latn pol_Latn nld_Latn rus_Cyrl ita_Latn por_Latn deu_Latn fra_Latn
spa_Latn
"}"

# Map the requested language codes to the exact subset codes that exist in
# FineWeb-2. After downloading, the directory is renamed to the requested
# name so the rest of the pipeline sees the expected language code.
actual_lang_for() {
    case "$1" in
        aze_Latn) echo "azj_Latn" ;;   # North Azerbaijani
        est_Latn) echo "ekk_Latn" ;;   # Standard Estonian
        lav_Latn) echo "lvs_Latn" ;;   # Standard Latvian
        mlg_Latn) echo "plt_Latn" ;;   # Plateau Malagasy
        msa_Latn) echo "zsm_Latn" ;;   # Standard Malay
        nep_Deva) echo "npi_Deva" ;;   # Nepali
        pus_Arab) echo "pbt_Arab" ;;   # Southern Pashto
        sqi_Latn) echo "als_Latn" ;;   # Tosk Albanian
        swa_Latn) echo "swh_Latn" ;;   # Swahili
        uzb_Latn) echo "uzn_Latn" ;;   # Northern Uzbek
        zho_Hans) echo "cmn_Hani" ;;   # Mandarin Chinese
        *) echo "$1" ;;
    esac
}

# Download one language subset. Uses `hf download` if available, else the
# legacy `huggingface-cli download` syntax.
download_lang() {
    local lang="$1"
    local actual
    actual="$(actual_lang_for "${lang}")"
    local download_dest="${DEST_ROOT}/data/${actual}"
    local final_dest="${DEST_ROOT}/data/${lang}"

    local includes=("data/${actual}/train/*.parquet" "data/${actual}/test/*.parquet")
    if [ "${INCLUDE_REMOVED}" = "1" ]; then
        includes+=("data/${actual}/removed/*.parquet")
    fi

    echo ">>> [${lang}] downloading FineWeb-2 subset '${actual}' -> ${DEST_ROOT}"
    if [ "${actual}" != "${lang}" ]; then
        echo "    (mapped ${lang} -> ${actual})"
    fi

    mkdir -p "${download_dest}"

    if [ "${HF_CLI}" = "hf" ]; then
        local args=(download "${REPO_ID}" --repo-type dataset
                    --local-dir "${DEST_ROOT}" --max-workers "${WORKERS}")
        for p in "${includes[@]}"; do args+=(--include "${p}"); done
        if ! hf "${args[@]}"; then
            echo "!!! [${lang}] download failed for subset ${actual}" >&2
            return 1
        fi
    else
        local args=(download "${REPO_ID}" --repo-type dataset
                    --local-dir "${DEST_ROOT}" --local-dir-use-symlinks False)
        for p in "${includes[@]}"; do args+=(--include "${p}"); done
        if ! huggingface-cli "${args[@]}"; then
            echo "!!! [${lang}] download failed for subset ${actual}" >&2
            return 1
        fi
    fi

    # Make sure the subset actually produced files (missing subsets result in
    # "Fetching 0 files" but the CLI still exits 0).
    if [ ! -d "${download_dest}" ] || [ -z "$(find "${download_dest}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "!!! [${lang}] download produced no files for ${download_dest}" >&2
        return 1
    fi

    # If we used a different FineWeb-2 code, rename the directory to the
    # requested language name.
    if [ "${actual}" != "${lang}" ]; then
        if [ -e "${final_dest}" ]; then
            if [ -d "${final_dest}" ] && [ -z "$(find "${final_dest}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
                rmdir "${final_dest}"
            else
                echo "!!! [${lang}] final destination ${final_dest} already exists; leaving ${download_dest} as-is" >&2
                return 1
            fi
        fi
        mv -v "${download_dest}" "${final_dest}"
    fi

    return 0
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main() {
    mkdir -p "${DEST_ROOT}" "${HF_HOME}"
    echo ">>> Using CLI: ${HF_CLI}"
    echo ">>> Destination: ${DEST_ROOT}"
    echo ">>> INCLUDE_REMOVED=${INCLUDE_REMOVED}  WORKERS=${WORKERS}"

    local total ok=0 fail=0
    # shellcheck disable=SC2086
    set -- ${LANGS}
    total="$#"
    local i=0
    for lang in "$@"; do
        i=$((i + 1))
        echo ">>> ($i/$total) ${lang}"
        if download_lang "${lang}"; then
            ok=$((ok + 1))
        else
            echo "!!! FAILED: ${lang}" >&2
            fail=$((fail + 1))
        fi
    done

    echo "================================================================"
    echo ">>> Done. success=${ok} failed=${fail} total=${total}"
    echo ">>> Data at: ${DEST_ROOT}/data"
    [ "${fail}" -eq 0 ]
}

main "$@"
