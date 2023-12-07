import streamlit as st
import pandas as pd

def main():
    st.title("Perhitungan Manual Excel")

    file_name = "perhitungan manual.xlsx"  

    uploaded_file = st.file_uploader(f"Unggah {file_name}", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"File {file_name} berhasil diunggah. Tampilan dataframe:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
