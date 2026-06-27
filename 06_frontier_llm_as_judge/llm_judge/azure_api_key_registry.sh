#!/bin/bash
# Maps API_KEY_ENV (.env variable names) to Azure endpoint, deployment, TPM, and RPM.
# Sourced by submit_llm_judge.sh; keys live only in .env, not here.

declare -A AZURE_API_ENDPOINT=(
    [AZURE_API_KEY]=""
)

declare -A AZURE_API_DEPLOYMENT=(
    [AZURE_API_KEY]="DeepSeek-V4-Flash"
)

declare -A AZURE_API_TPM=(
    [AZURE_API_KEY]=250000
)

declare -A AZURE_API_RPM=(
    [AZURE_API_KEY]=250
)

# resolve_azure_from_api_key_env NAME
# Sets RESOLVED_ENDPOINT, RESOLVED_DEPLOYMENT, RESOLVED_TPM, and RESOLVED_RPM; returns 0 on hit, 1 if unknown.
resolve_azure_from_api_key_env() {
    local env_name="$1"
    if [[ -n "${AZURE_API_ENDPOINT[$env_name]:-}" ]]; then
        RESOLVED_ENDPOINT="${AZURE_API_ENDPOINT[$env_name]}"
        RESOLVED_DEPLOYMENT="${AZURE_API_DEPLOYMENT[$env_name]}"
        RESOLVED_TPM="${AZURE_API_TPM[$env_name]}"
        RESOLVED_RPM="${AZURE_API_RPM[$env_name]}"
        return 0
    fi
    return 1
}
