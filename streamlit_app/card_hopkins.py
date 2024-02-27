import streamlit as st
import pandas as pd
import plotly.express as px

def page():
    """
    Add Cardinality indicator and Hopkins Statistic and show correlations to other measures and PNG Size
    """
    card_tab, hop_tab, cor_tab = st.tabs(['Cardinality', 'Hopkins Statistic', 'Correlations'])

    with card_tab:
        st.markdown(open('markdown/measures/card.md','r').read())
    with hop_tab:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(open('markdown/measures/hopkins_description.md','r').read())
        with col2:
            st.markdown(open('markdown/measures/hopkins_adaptation.md','r').read())
    with cor_tab:
        st.markdown('Then, correlations heatmap, including $Card$ and $Hopkins$ :')
        df = pd.read_csv('./precomputed/train.csv')
        cor = df.corr().abs()
        fig = px.imshow(cor.fillna(0)
                        .sort_values('PNG Size', ascending=False, axis=0)
                        .sort_values('PNG Size', ascending=False, axis=1), color_continuous_scale='blues')
        fig.update_coloraxes(showscale=False)
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.components.v1.html(fig.to_html(include_mathjax='cdn'), height=500)