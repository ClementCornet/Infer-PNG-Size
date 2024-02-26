import streamlit as st
import plotly.express as px


# Local Imports
from helpers.images import get_images

def page():
    """
    Simple introduction to this work.
    """
    st.title('Estimating Compressed PNG Size')

    col1, col2 = st.columns([1,1])

    # PNG & CIFAR-100
    with col1:
        st.markdown(open('markdown/motivation/png_format.md','r').read())
        st.markdown(open('markdown/motivation/methods_overview.md','r').read())
        
    # Methods Overview
    with col2:
        st.markdown(open('markdown/motivation/cifar100.md','r').read())

        img = get_images()[0]
        fig = px.imshow(img, title=f'{" "*60}Sample Image')
        fig.update_annotations(margin=dict(r=0,t=0,b=0,l=0))
        st.plotly_chart(fig, use_container_width=True)