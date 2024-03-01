import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Lasso
import plotly.express as px
from sklearn.preprocessing import StandardScaler

from helpers.images import get_images


def page():

    """
    Build a simple Lasso Regressor, using Sparsity Measurements, to predict PNG Size
    """
    
    im_test = get_images('test')
    
    df_train = pd.read_csv('./precomputed/train.csv').clip(lower=-1e300, upper=1e300)
    df_test = pd.read_csv('./precomputed/test.csv').clip(lower=-1e300, upper=1e300)

    X_train = df_train.drop('PNG Size', axis=1)\
                .drop('$Hopkins$', axis=1)\
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
    X_test = df_test.drop('PNG Size', axis=1)\
                .drop('$Hopkins$', axis=1)\
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
    
    cols = X_train.columns
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)

    y_train = df_train['PNG Size']
    y_test = df_test['PNG Size']

    col1, col2 = st.columns([1,0.7])

    with col1:
        st.markdown(' Predict PNG Size using sparsity measurements. Lasso Regression, using $\lambda=1$.')

        model = Lasso(alpha=1)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        #st.markdown(f"- $R^2 =  {{{model.score(X_test, y_test):_.3f}}}$")
        #st.markdown(f"- $MAE\% = {{{mean_absolute_percentage_error(pred, y_test):_.3f}}}$")

        fig = px.scatter(
            x=pred, y=y_test, 
            trendline='ols', trendline_color_override='red',
            labels={
                     "x": "Predicted",
                     "y": "PNG Size",
                 },
            title=f'R² = {model.score(X_test, y_test):_.3f}\t\t|\t\tMAE% = {100*mean_absolute_percentage_error(pred, y_test):_.3f}%',
        )
        st.plotly_chart(fig, use_container_width=True)
        #st.components.v1.html(fig.to_html(include_mathjax='cdn'),height=500)
        st.write("Some very large error subsist, is there a pattern in overestimations/underestimations ?")

        #fig = px.histogram(pred/y_test)
        #st.plotly_chart(fig)


    with col2:
        idx_over = pd.Series((pred-y_test)).idxmax()
        idx_under = pd.Series((pred-y_test)).idxmin()

        f1_title = f'{" "*30}Predicted : {int(pred[idx_over])}, Real : {y_test[idx_over]}'
        f2_title = f'{" "*30}Predicted : {int(pred[idx_under])}, Real : {y_test[idx_under]}'

        f1 = px.imshow(im_test[idx_over], title=f1_title)
        f2 = px.imshow(im_test[idx_under], title=f2_title)
        f1.update_xaxes(showticklabels=False)
        f1.update_yaxes(showticklabels=False)
        f1.update_annotations(margin=dict(b=0,t=0))
        f2.update_xaxes(showticklabels=False)
        f2.update_yaxes(showticklabels=False)

        t1, t2 = st.tabs(['Overestimation', 'Underestimation'])

        with t1:
            st.write("Seems to have very few different colors / slight variations : could it explain the low compressed size?")
            st.plotly_chart(f1, use_container_width=True)
        with t2:
            st.write("Contains cluster-like structures :  could those structures bias the predictor?")
            st.plotly_chart(f2, use_container_width=True)