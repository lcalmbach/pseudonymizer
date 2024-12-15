import streamlit as st
import pandas as pd
import pseudonymizer
import json

file_path = "./data/file2pseudomize.xlsx"
json_file_path = "./data/config.json"


def main():
    if st.button("Create Template"):
        pseudonymizer.generate_json_template(file_path)

    st.title("👤 Pseudonymizer")
    uploaded_file = st.file_uploader("Choose a file")
    uploaded_config = st.file_uploader("Choose a config file")
    if uploaded_config:
        with open(json_file_path, "wb") as f:
            f.write(uploaded_config.getvalue())
    if uploaded_file:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    if uploaded_file and uploaded_config:
        if st.button("👤 Pseudonymize File"):
            with st.spinner("Pseudonymizing..."):
                p = pseudonymizer.DataMasker(file_path, json_file_path)
                p.pseudonymize()


if __name__ == "__main__":
    main()
