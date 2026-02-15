import streamlit as st

st.title("📚 Quality Taxonomy Guide")
st.markdown("This tool uses the **MQM (Multidimensional Quality Metrics)** framework to standardize human evaluation.")

# Data structure for the taxonomy
taxonomy_data = {
    "Mistranslation": {
        "brief": "The target text does not accurately reflect the meaning of the source text.",
        "detailed": "Includes false friends, misinterpretation of technical relationships, or incorrect negation. The core message is lost or distorted."
    },
    "Omission": {
        "brief": "Content present in the source is missing in the target.",
        "detailed": "Significant information (names, numbers, or entire clauses) is left out, making the translation incomplete."
    },
    "Untranslated": {
        "brief": "Source language text is left as-is in the target.",
        "detailed": "Common in low-resource MT where the model 'gives up' and passes the source word through. This is distinct from intentional loanwords."
    },
    "MT Hallucination": {
        "brief": "The translation contains content that has no basis in the source.",
        "detailed": "The model generates fluent-sounding but completely unrelated text, often a major risk in safety-critical domains."
    }
}

st.subheader("Issue Categories")

for category, content in taxonomy_data.items():
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        col1.markdown(f"**{category}**")
        col2.write(content["brief"])
        
        with st.expander("View Detailed Definition"):
            st.info(content["detailed"])

st.divider()
st.info("💡 **Tip for Annotators:** When in doubt, prioritize 'Accuracy' issues over 'Style' issues unless the style renders the text unusable.")