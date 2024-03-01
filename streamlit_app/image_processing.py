import streamlit as st
from scipy import ndimage as ndi
from skimage import data
from skimage import color
import plotly.express as px
import numpy as np
import pandas as pd

def page():
    gaussian_tab, greyop_tab, cor_tab = st.tabs(['Difference of Gaussians', 'Grayscale Opening', 'Correlations'])

    with gaussian_tab:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(open('markdown/image_processing/gaussian_filtering.md','r').read())

        with col2:
            astro = color.rgb2gray(data.astronaut())
            astro_dog = ndi.gaussian_filter(astro, .5) - ndi.gaussian_filter(astro, 1)
            fig = px.imshow(astro_dog, color_continuous_scale='gray', title=f'{" "*75}DoG example')
            st.plotly_chart(fig)

    with greyop_tab:
        col1, _, col2 = st.columns([2, 0.5, 1.5])
        with col1:
            st.markdown(open('markdown/image_processing/grayscale_opening.md','r').read())
        with col2:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Grayscale_Morphological_Erosion.gif/330px-Grayscale_Morphological_Erosion.gif",
                width=250,
                caption='Erosion'
            )
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Grayscale_Morphological_Dilation.gif/330px-Grayscale_Morphological_Dilation.gif",
                width=250,
                caption='Dilation'
            )

    with cor_tab:
        st.markdown('Then, correlations heatmap, including those new indicators :')
        df = pd.read_csv('./precomputed/train.csv')
        cor = df.corr().abs()
        fig = px.imshow(cor.fillna(0)
                        .sort_values('PNG Size', ascending=False, axis=0)
                        .sort_values('PNG Size', ascending=False, axis=1), color_continuous_scale='blues')
        fig.update_coloraxes(showscale=False)
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.components.v1.html(fig.to_html(include_mathjax='cdn'), height=500)