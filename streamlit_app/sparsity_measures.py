import streamlit as st
import pandas as pd
import plotly.express as px

# Local Imports
from helpers.table import measures_table 

def page():
    """
    Sparsity Measures page : list measures, and display correlations with PNG Size
    """
    st.title('Sparsity Measures')

    col1, col2 = st.columns([1,1.5])
    with col1:
        st.markdown(open('markdown/measures/sparse_measures.md','r').read())

    #with st.sidebar:
    #    epsilon_l0eps = param_widget('$\epsilon$ for $\ell_\epsilon^0$', 0.0, 1.0, 0.1)
    #    p_lp = param_widget('$p$ for $\ell^p$', 0.0, 5.0, 2.0)
    #    a_tanh = param_widget('$a$ for $tanh_{a,b}$', 0.1, 10.0, 2.0)
    #    b_tanh = param_widget('$b$ for $tanh_{a,b}$', 0.0, 10.0, 2.0)
    #    p_lp_neg = param_widget('$p$ for $\ell^p_-$', -5.0, 0.0, -2.0)

    with col2:
        #df = measures_table(
        #    subset='train',
        #    epsilon_l0eps=epsilon_l0eps,
        #    p_lp=p_lp,
        #    a_tanh=a_tanh,
        #    b_tanh=b_tanh,
        #    p_lp_neg=p_lp_neg
        #)
        df = pd.read_csv('./precomputed/train.csv')
        df2 = df.drop('$Hopkins$', axis=1)\
                    .drop('$Card$', axis=1)\
                    .drop('$Card_{raw}$', axis=1)\
                    .drop('$Card_{raw}^{mono}$', axis=1)\
                    .drop('$GO-\ell^0$', axis=1)\
                    .drop('$GO-\ell^1$', axis=1)\
                    .drop('$GO-\ell^2$', axis=1)\
                    .drop('$GO-\ell^H$', axis=1)\
                    .drop('$DoG-\ell^0$', axis=1)\
                    .drop('$DoG-\ell^1$', axis=1)\
                    .drop('$DoG-\ell^2$', axis=1)\
                    .drop('$DoG-\ell^H$', axis=1)\
                    .drop('$GO-\ell^0_{raw}$', axis=1)\
                    .drop('$GO-\ell^1_{raw}$', axis=1)\
                    .drop('$GO-\ell^2_{raw}$', axis=1)\
                    .drop('$GO-\ell^H_{raw}$', axis=1)\
                    .drop('$DoG-\ell^0_{raw}$', axis=1)\
                    .drop('$DoG-\ell^1_{raw}$', axis=1)\
                    .drop('$DoG-\ell^2_{raw}$', axis=1)\
                    .drop('$DoG-\ell^H_{raw}$', axis=1)
        cor = df2.corr().abs()
        fig = px.imshow(cor.fillna(0)
                        .sort_values('PNG Size', ascending=False, axis=0)
                        .sort_values('PNG Size', ascending=False, axis=1), color_continuous_scale='blues')
        fig.update_coloraxes(showscale=False)
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.components.v1.html(fig.to_html(include_mathjax='cdn'),height=500)
        

def param_widget(label, minval, maxval, placeholder):
    """
    Slider aligned horizontally with its label.
    Might cause streamlit warnings
    """
    subcol1, subcol2 = st.columns([1,4])
    with subcol1:
        st.markdown('')
        st.markdown('')
        st.markdown(f"\n{label}")
    with subcol2:
        res = st.slider('.', minval, maxval, placeholder, label_visibility='hidden')
        return res