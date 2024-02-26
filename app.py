# Standard Streamlit imports
import streamlit as st
st.set_page_config(page_title='Sparse #4', layout="wide")
import warnings
warnings.filterwarnings('ignore')

# Local Imports
import motivation

with st.sidebar:
    st.title('Sparse Models Project #4')

    choice = st.selectbox('Choice',[
        'Motivation',
        'Sparsity Measurements',
        'First Predictor',
        'New Measures'
    ])

if choice == 'Motivation':
    motivation.page()