import streamlit as st

st.title("🛡️ FineOPUS Data Integrity & Validation Portal")

st.markdown("""
### High-Stakes Linguistic Validation
Welcome to the data integrity portal. You are tasked with performing a rigorous **Linguistic Validation** of the **FineOPUS-Original** dataset. 

The quality of this data is the primary safeguard for the safety and accuracy of next-generation AI systems. Your role is to serve as the final authority, ensuring that machine-generated translations meet the stringent benchmarks required for critical deployment. To maintain the scientific integrity of the dataset, adherence to the Validation Framework is non-negotiable.
""")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🕵️‍♂️ Validation Protocol")
    st.markdown("""
    - **Cross-Verification:** Conduct a high-granularity comparison between the source text and the machine-generated target.
    - **Failure Diagnostics:** Identify and isolate linguistic anomalies using the standardized MQM framework.
    - **Impact Assessment:** Gauge the severity of deviations (Minor, Major, or Critical) on the reliability of the data.
    - **Integrity Certification:** Conclude your review by exporting the validated records in JSONL format for final data ingestion.
    """)

with col2:
    st.warning("""
    ⚠️ **Integrity Warning:** Compromised validation leads to systemic model failure. If a record fulfills all accuracy and fluency benchmarks, certify it as **'Verified (None)'**.
    """)

st.divider()

# Guidance Section
st.subheader("📚 Phase 1: Review Validation Standards")
st.info("""
Before granting certification, you must internalize the **Linguistic Validation Standards**. 
Precise distinction between 'Mistranslation' and 'Hallucination' is the cornerstone of high-integrity data.
""")

if st.button("Review Validation Standards", type="secondary", use_container_width=True):
    st.switch_page("taxonomy.py")

st.divider()

# Execution Section
st.subheader("🚀 Phase 2: Commencing Data Validation")
st.write("By entering the portal, you acknowledge your role in certifying the accuracy of this critical dataset.")

if st.button("Access Validation Tool", type="primary", use_container_width=True):
    st.switch_page("annotator.py")

st.caption("The Validation Tool can also be accessed directly via the sidebar navigation.")