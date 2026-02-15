# 🛡️ FineOPUS Data Integrity & Validation Portal

This application is a specialized **Linguistic Quality Assurance (LQA)** tool designed to stream, audit, and certify translation data from the `MaLA-LM/FineOPUS-Original` dataset. It transforms the standard annotation process into a rigorous **Validation Protocol**.

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have **Python 3.9+** installed. You will need the following libraries:

```bash
pip install streamlit datasets pandas pyarrow

```

### 2. Launching the Portal

Clone the repository and run the entry point file:

```bash
streamlit run app.py

```

---

## 🛠️ Application Architecture

The portal is structured into three specialized modules to ensure data integrity:

1. **Project Home (`home.py`):** The landing zone that defines the mission scope and audit protocols.
2. **Validation Standards (`taxonomy.py`):** A comprehensive guide to the **MQM (Multidimensional Quality Metrics)** framework used to categorize linguistic anomalies.
3. **Validation Tool (`annotator.py`):** The execution environment for side-by-side data verification and batch certification.

---

## 📖 Operational Guide

### Phase 1: Internalizing Standards

Before commencing an audit, navigate to the **Validation Standards** page.

* Review the **Brief Descriptions** for common failure points like *Mistranslation* or *MT Hallucinations*.
* Toggle the **Detailed Version** to view edge cases and specific linguistic examples.
* **Goal:** Ensure your diagnostics are consistent with global LQA benchmarks.

### Phase 2: Data Streaming & Configuration

In the **Validation Tool** sidebar:

* **Select Language Pair:** Choose from the standard `xxx_Yyyy-zzz_Yyyy` configurations (e.g., `deu_Latn-eng_Latn`).
* **Sample Count:** Define the number of records to stream. The tool uses **Zero-Download Streaming**, pulling data directly from the cloud without utilizing local disk space.

### Phase 3: Executing the Audit

For each record in the batch:

1. **Compare:** Analyze the `source_text` against the `target_text`.
2. **Diagnose:** Select the **Issue Type** from the dropdown if a deviation is found.
3. **Assess Severity:** Categorize the impact as **Minor** (fluency), **Major** (accuracy), or **Critical** (safety/hallucination).
4. **Certify:** Click **"Save Batch"** to commit your findings to the session memory.

### Phase 4: Data Export & Ingestion

Once a batch is certified, the **Export** section will activate:

* **JSONL Export:** Generates a line-delimited JSON file where each line is a dictionary containing original data + your validation metadata.
* **Parquet Export:** Generates a compressed columnar file for large-scale machine learning pipelines.

---

## 📂 Exported Data Schema

The exported `.jsonl` files follow this structure:

| Field | Description |
| --- | --- |
| `source_text` | The original input text (from FineOPUS). |
| `target_text` | The machine-generated translation (from FineOPUS). |
| `lqa_issue` | The diagnostic result (e.g., Mistranslation, Omission). |
| `lqa_severity` | The quantified impact of the error. |
| `pair_id` | The language configuration folder used. |

---

## ⚠️ Security & Persistence Note

This portal utilizes **Streamlit Session State**. Your progress is stored in the browser session.

* **Do not refresh the page** mid-batch, as unsaved certifications will be lost.
* Always **Download the JSONL** before closing the application or switching to a new language pair if you wish to maintain separate logs.


---

## 🚧 TODO
[] Fix dataset config and loading language pairs
[] Improve efficiency 