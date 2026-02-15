import streamlit as st

# Define the pages
home_page = st.Page("home.py", title="Home", icon="🏠", default=True)
annotate_page = st.Page("annotator.py", title="Annotation Tool", icon="🎯")
taxonomy_page = st.Page("taxonomy.py", title="Quality Taxonomy", icon="📚")

# Create Navigation
pg = st.navigation([home_page, taxonomy_page, annotate_page])

# Global Page Config
st.set_page_config(page_title="LQA Hub", layout="wide")

# Run Navigation
pg.run()