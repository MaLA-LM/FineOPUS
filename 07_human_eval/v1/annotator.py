import streamlit as st
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
import io, json

HF_REPO = "MaLA-LM/FineOPUS-Original"

# State management
if "annotations" not in st.session_state:
    st.session_state.annotations = {}

@st.cache_data
def get_configs():
    return get_dataset_config_names(HF_REPO)

def stream_data(config, n):
    ds = load_dataset(HF_REPO, config, streaming=True)
    # Extract the first split from IterableDatasetDict
    if isinstance(ds, dict):
        ds = ds[list(ds.keys())[0]]
    return pd.DataFrame(list(ds.take(n)))

# Sidebar UI
st.sidebar.header("Data Stream Settings")
configs = get_configs()
print(f"Found {len(configs)} language pairs:")
for config in configs[:10]:  # Show first 10
    print(f" - {config}")
    
selected_pair = st.sidebar.selectbox("Language Pair (xxx_Yyyy-zzz_Yyyy)", configs)
n_samples = st.sidebar.slider("Sample Count", 5, 100, 5)

df = stream_data(selected_pair, n_samples)

st.header(f"🎯 Annotating {selected_pair}")

# Annotation Form
with st.form("batch_form"):
    updates = {}
    for idx, row in df.iterrows():
        c1, c2, c3 = st.columns([2, 2, 1.5])
        c1.markdown(f"**Source**\n\n{row.get('source_text', 'N/A')}")
        c2.markdown(f"**Target**\n\n{row.get('target_text', 'N/A')}")
        
        with c3:
            issue = st.selectbox("Issue", ["None", "Mistranslation", "Omission", "Untranslated", "Hallucination"], key=f"i_{idx}")
            sev = st.select_slider("Severity", ["Minor", "Major", "Critical"], key=f"s_{idx}")
            updates[idx] = {"issue": issue, "severity": sev}
        st.divider()
    
    if st.form_submit_button("Save Batch"):
        st.session_state.annotations.update(updates)
        st.success("Batch saved to session!")

# Export Logic
if st.session_state.annotations:
    st.subheader("📤 Export Results")
    export_data = [ {**row.to_dict(), **st.session_state.annotations.get(idx, {})} for idx, row in df.iterrows() ]
    
    jsonl_output = "\n".join([json.dumps(r, ensure_ascii=False) for r in export_data])
    st.download_button("Download JSONL", data=jsonl_output, file_name=f"{selected_pair}_lqa.jsonl")