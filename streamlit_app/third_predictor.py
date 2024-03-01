import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Lasso
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import sklearn

from helpers.images import get_images


def page():

    """
    Build a simple Lasso Regressor, using Sparsity Measurements, Cardinality and Hopkins, to predict PNG Size
    """
    
    im_test = get_images('test')
    
    df_train = pd.read_csv('./precomputed/train.csv').clip(lower=-1e300, upper=1e300)
    df_test = pd.read_csv('./precomputed/test.csv').clip(lower=-1e300, upper=1e300)

    scaler = sklearn.preprocessing.StandardScaler()

    cols = df_train.drop('PNG Size', axis=1).columns

    X_train = scaler.fit_transform(df_train.drop('PNG Size', axis=1))
    X_test = scaler.transform(df_test.drop('PNG Size', axis=1))
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = pd.DataFrame(X_test, columns=cols)

    y_train = df_train['PNG Size']
    y_test = df_test['PNG Size']

    col1, col2 = st.columns([1,0.7])

    with col1:
        st.markdown('''Predict PNG Size using sparsity measurements. Lasso Regression, using $\lambda=0.5$.
                    This time, using the image processing metrics ''')

        model = Lasso(alpha=0.5)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

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


    with col2:
        #idx_over = pd.Series((pred-y_test)).idxmax()
        #idx_under = pd.Series((pred-y_test)).idxmin()
#
        #f1_title = f'{" "*30}Predicted : {int(pred[idx_over])}, Real : {y_test[idx_over]}'
        #f2_title = f'{" "*30}Predicted : {int(pred[idx_under])}, Real : {y_test[idx_under]}'
#
        #f1 = px.imshow(im_test[idx_over], title=f1_title)
        #f2 = px.imshow(im_test[idx_under], title=f2_title)
        #f1.update_xaxes(showticklabels=False)
        #f1.update_yaxes(showticklabels=False)
        #f1.update_annotations(margin=dict(b=0,t=0))
        #f2.update_xaxes(showticklabels=False)
        #f2.update_yaxes(showticklabels=False)

        #t1, t2 = st.tabs(['Overestimation', 'Underestimation'])
#
        #with t1:
        #    st.info('Voir si ça aide de considérer si une image est +/- grise? Sinon, remplacer colonne par feature importance/histogram')
        #    #st.write("Seems to have very few different colors / slight variation : could it explain the low compressed size?")
        #    st.plotly_chart(f1, use_container_width=True)
        #with t2:
        #    #st.write("Contains cluster-like structures :  could those structures bias the predictor?")
        #    st.plotly_chart(f2, use_container_width=True)

        t1, t2 = st.tabs(['Features Importance', 'Error Histogram'])
        with t1:
            coeff = pd.DataFrame()
            coeff['Feature'] =  model.feature_names_in_
            coeff['coeff'] = model.coef_
            coeff.sort_values('coeff')
            fig = px.bar(coeff[coeff['coeff']**2>2], y='Feature', x='coeff', orientation='h', labels={
                'Feature' : '',
                'coeff' : 'Coefficients'
            }, title='Lasso Regression Coefficients')
            fig.update_traces(marker_color='#0068c9')
            st.components.v1.html(fig.to_html(include_mathjax='cdn'),height=500)
        
            

        with t2:
            fig = px.histogram(pred/y_test - 1, title='Error histogram')
            fig.update_layout(showlegend=False)
            fig.update(layout_showlegend=False)
            st.plotly_chart(fig, use_container_width=True)



        st.write('A partir d\'ici c\'est le bordel')
        import xgboost as xgb
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        from sklearn.model_selection import GridSearchCV
        # set up our search grid
        param_grid = {"max_depth":    [5],
                    "n_estimators": [2250],
                    "learning_rate": [0.0375]}

        # try out every combination of the above values
        regressor=xgb.XGBRegressor(eval_metric='rmsle', missing=np.inf)

        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
        #X_train = scaler.transform(X_train)
        #X_test = scaler.fit_transform(X_test)
        search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)
        st.write(search.best_params_)
        model=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                            n_estimators  = search.best_params_["n_estimators"],
                            max_depth     = search.best_params_["max_depth"],
                            eval_metric='rmsle',
                            missing=np.inf) 
        #model = xgb.XGBRegressor(eval_metric='rmsle', missing=np.inf)
        model.fit(pd.DataFrame(X_train), y_train)
        pred = model.predict(X_test)
        fig = px.scatter(
            x=pred, y=y_test, 
            trendline='ols', trendline_color_override='red',
            labels={
                        "x": "Predicted",
                        "y": "PNG Size",
                    },
            title=f'R² = {model.score(pd.DataFrame(X_test), y_test):_.3f}\t\t|\t\tMAE% = {100*mean_absolute_percentage_error(pred, y_test):_.3f}%',
        )
        st.write('sisi c\'est ici youpi')
        st.plotly_chart(fig, use_container_width=True)
        from xgboost import plot_importance

        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')
        plt.rcParams.update({'font.size': 16})

        fig, ax = plt.subplots(figsize=(12,6))
        plot_importance(model, max_num_features=15, ax=ax)
        st.pyplot(fig)

        fig = px.histogram(pred/y_test)
        st.plotly_chart(fig)


        #idx_over = pd.Series((pred-y_test)).idxmax()
        #idx_under = pd.Series((pred-y_test)).idxmin()
#
        #f1_title = f'{" "*30}Predicted : {int(pred[idx_over])}, Real : {y_test[idx_over]}'
        #f2_title = f'{" "*30}Predicted : {int(pred[idx_under])}, Real : {y_test[idx_under]}'
#
        #f1 = px.imshow(im_test[idx_over], title=f1_title)
        #f2 = px.imshow(im_test[idx_under], title=f2_title)
        #f1.update_xaxes(showticklabels=False)
        #f1.update_yaxes(showticklabels=False)
        #f1.update_annotations(margin=dict(b=0,t=0))
        #f2.update_xaxes(showticklabels=False)
        #f2.update_yaxes(showticklabels=False)
#
        #t1, t2 = st.tabs(['Overestimation', 'Underestimation'])
#
        #with t1:
        #    st.write("Seems to have very few different colors / slight variations : could it explain the low compressed size?")
        #    st.plotly_chart(f1, use_container_width=True)
        #with t2:
        #    st.write("Contains cluster-like structures :  could those structures bias the predictor?")
        #    st.plotly_chart(f2, use_container_width=True)
