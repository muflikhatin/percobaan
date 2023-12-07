import pandas as pd
import streamlit as st
import numpy as np
import nltk
from sklearn.utils.validation import joblib

# st.markdown("# 1. Information")
# create content


def main():
    st.title("Halaman Informasi")
    st.header("Analisis sentimen")
    st.container()
    st.write("""
            * Analisis sentimen adalah proses memahami dan mengevaluasi opini atau sentimen dari teks, seperti ulasan, tweet, atau posting media sosial lainnya. 
            * Tujuan utamanya adalah untuk menentukan apakah suatu teks memiliki sentimen positif, negatif, atau netral
            """)

    st.header("Informasi Data")

    # Crowling data
    st.write("""
    * Data diperoleh dari keggel https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
    * Data yang di peroleh memiliki 10 parameter namun yang digunakan hanya tweet text
    * Data diperoleh pada tahun 2022
    """)
    
    st.markdown("## Formula TF-IDF")

    st.latex(r'''
    W1 = tf1 \times \log \frac{D}{df1}
    ''')

    st.write("""
    KETERANGAN:
    - W1 adalah bobot dari term (kata atau token) pertama dalam dokumen.
    - tf1 adalah frekuensi kemunculan term pertama dalam dokumen (Term Frequency).
    - df1 adalah jumlah dokumen yang berisi term pertama (Document Frequency).
    - D adalah total jumlah dokumen dalam korpus atau koleksi dokumen yang sedang diproses.
    - log adalah fungsi logaritma yang umumnya menggunakan basis 10 atau basis lainnya tergantung pada implementasi yang digunakan
    """)

    st.markdown("## Formula Backpropagation")

    st.latex(r'''
    \delta_j = \frac{\partial E}{\partial a_j} \cdot \frac{\partial a_j}{\partial z_j} = \frac{\partial E}{\partial a_j} \cdot \sigma'(z_j)
    ''')

    st.write("""
    KETERANGAN:
    - \delta_j adalah kesalahan (error) dari neuron ke-j dalam jaringan.
    - E adalah fungsi kesalahan (error function).
    - a_j adalah output dari neuron ke-j.
    - z_j adalah input tertimbang (weighted input) dari neuron ke-j.
    - \sigma'(z_j) adalah turunan dari fungsi aktivasi \sigma terhadap input tertimbang z_j.
    """)
    
    st.markdown("## Formula LSTM (Long Short-Term Memory)")

    st.latex(r'''
    \begin{align*}
    i_t &= \sigma(W_{ii}x_t + b_{ii} + W_{hi}h_{(t-1)} + b_{hi}) \\
    f_t &= \sigma(W_{if}x_t + b_{if} + W_{hf}h_{(t-1)} + b_{hf}) \\
    g_t &= \tanh(W_{ig}x_t + b_{ig} + W_{hg}h_{(t-1)} + b_{hg}) \\
    o_t &= \sigma(W_{io}x_t + b_{io} + W_{ho}h_{(t-1)} + b_{ho}) \\
    c_t &= f_t \odot c_{(t-1)} + i_t \odot g_t \\
    h_t &= o_t \odot \tanh(c_t)
    \end{align*}
    ''')

    st.write("""
    KETERANGAN:
    - $i_t$, $f_t$, $g_t$, dan $o_t$ adalah vektor sigmoid yang merepresentasikan gate pada sel LSTM (input, forget, memory, dan output gate).
    - $c_t$ adalah vektor sel (cell state) pada waktu $t$.
    - $h_t$ adalah vektor output pada waktu $t$.
    - $x_t$ adalah input pada waktu $t$.
    - $h_{(t-1)}$ dan $c_{(t-1)}$ adalah output dan sel pada waktu sebelumnya.
    - $W$ dan $b$ adalah bobot dan bias yang diterapkan pada input, hidden state, dan gate.
    - $\sigma$ adalah fungsi sigmoid, dan $\odot$ merupakan operasi perkalian elemen-wise (element-wise multiplication).
    """)

if __name__ == "__main__":
    main()
