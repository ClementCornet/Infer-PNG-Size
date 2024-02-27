# Standard Streamlit imports
import streamlit as st
st.set_page_config(page_title='Sparse #4', layout="wide")
import warnings
warnings.filterwarnings('ignore')

# Local Imports
from streamlit_app import motivation, sparsity_measures, first_predictor, card_hopkins, second_predictor

with st.sidebar:
    st.title('Sparse Models Project #4')

    choice = st.selectbox('Choice',[
        'Motivation',
        'Sparsity Measures',
        'First Predictor',
        'New Measures',
        'Second Predictor'
    ])

if choice == 'Motivation':
    motivation.page()

if choice == 'Sparsity Measures':
    sparsity_measures.page()

if choice == 'First Predictor':
    first_predictor.page()

if choice == 'New Measures':
    card_hopkins.page()

if choice == 'Second Predictor':
    second_predictor.page()