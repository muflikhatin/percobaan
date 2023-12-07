# app.py
import streamlit as st
from main import main as main_page
from psd import main as psd_page
from excel import main as excel_page

# Sidebar untuk navigasi
selected_page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Halaman Utama", "Halaman Analisis Data", "Halaman Perhitungan Manual"])

# Menampilkan konten halaman yang dipilih
if selected_page == "Halaman Utama":
    main_page()
elif selected_page == "Halaman Analisis Data":
    psd_page()
elif selected_page == "Halaman Perhitungan Manual":
    excel_page()
# elif selected_page == "Halaman Term Frequency":
#     term_frequency_page()
# elif selected_page == "Halaman K Means":
#     K_Means_page()
# elif selected_page == "Halaman lda knn":
#     lda_knn_page()
# elif selected_page == "Halaman closeness centrality crawling":
#     closeness_centrality_crawling_page()
# elif selected_page == "Halaman closeness centrality":
#     closeness_centrality_page()
# else:
#     topic_modelling_page()
