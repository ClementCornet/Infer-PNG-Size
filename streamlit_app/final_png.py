import streamlit as st
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error

from helpers.measures import *
from helpers.images import paeth


def page():
    """
    XGBoost Regressor with all features + test on photos from user camera
    """

    t1, t2 = st.tabs(['XGBoost', 'Test with Camera'])
    
    scaler = StandardScaler()

    df_train = pd.read_csv('./precomputed/train.csv')
    df_test = pd.read_csv('./precomputed/test.csv')

    X_train = df_train.drop('PNG Size', axis=1)
    X_test = df_test.drop('PNG Size', axis=1)
    y_train = df_train['PNG Size']
    y_test = df_test['PNG Size']

    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.fit_transform(X_test)

    model = xgb.XGBRegressor(
        eval_metric='rmsle', 
        missing=np.inf,
        learning_rate = 0.0375,
        n_estimators  = 2250,
        max_depth     = 5
    )

    model.fit(X_train, y_train)

    with t1:
        col1, col2 = st.columns([1,1])

        with col1:
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
            st.write('Now, our model achieves satifying results over most images of our dataset')
        with col2:
            fig = px.histogram(pred/y_test - 1, labels={
                'value' : 'Relative Error'
            }, title='Relative Error Histogram')
            st.plotly_chart(fig, use_container_width=True)
            #st.write((pred/y_test - 1).describe())
            #st.write((pred/y_test - 1).quantile(0.1))
            #st.write((pred/y_test - 1).quantile(0.9))
            abs_rel_error = np.abs((pred/y_test - 1))
            st.write(f'Proportion of test images with less than 5% of absolute relative error : {100*(abs_rel_error<=0.05).mean():_.2f}%')

    with t2:
        from PIL import Image
        import cv2

        col1, col2 = st.columns([1, 1])

        with col1:
            img_file_buffer = st.camera_input("Take a picture")
            #st.write(df_train.columns)

            
        with col2:
            if img_file_buffer is not None:
                # To read image file buffer as a PIL Image:
                img = Image.open(img_file_buffer)

                # To convert PIL Image to numpy array:
                img_array = np.array(img)

                # Check the type of img_array:
                # Should output: <class 'numpy.ndarray'>
                
                im_raw = cv2.resize(img_array, dsize=(32, 32), interpolation=cv2.INTER_CUBIC).reshape((32,32,3))#.transpose(2,1,0)
                image_png_size = get_png_size(im_raw)
                #st.write(f'PNG Size : {image_png_size}')

                #st.write(im_raw.shape)
                im_paeth = paeth(im_raw.transpose(2,1,0))

                #st.write(im_paeth.shape)
                
                #st.info('Get all features of image + GET PAETH')
                
                
                image_features = np.array([
                    [image_hopkins(im_raw)],
                    [shannon_entropy(im_paeth)],
                    [modified_shannon_entropy(im_paeth)],
                    [l0_norm(im_paeth)],
                    [l1_norm(im_paeth)],
                    [l2_l1_ratio(im_paeth)],
                    [sparse_log(im_paeth)],
                    [kurtosis_4(im_paeth)],
                    [gaussian_entropy(im_paeth)],
                    [hoyer(im_paeth)],
                    [gini(im_paeth)],
                    [gini(im_raw)],
                    [card_image(im_paeth)],
                    [card_image(im_raw)],
                    [card_image_mono(im_raw)],
                    [dog_l0(im_raw)],
                    [dog_l1(im_raw)],
                    [dog_l2(im_raw)],
                    [dog_hs(im_raw)],
                    [greyopening_l0(im_raw)],
                    [greyopening_l1(im_raw)],
                    [greyopening_l2(im_raw)],
                    [greyopening_hs(im_raw)],
                    [dog_l0(im_paeth)],
                    [dog_l1(im_paeth)],
                    [dog_l2(im_paeth)],
                    [dog_hs(im_paeth)],
                    [greyopening_l0(im_paeth)],
                    [greyopening_l1(im_paeth)],
                    [greyopening_l2(im_paeth)],
                    [greyopening_hs(im_paeth)],
                    [sparse_tanh(im_paeth, 0.5, 2)],
                    [l0_epsilon(im_paeth, 0.005)],
                    [lp_norm(im_paeth, 2)],
                    [lp_neg(im_paeth, 0.5)]
                ]).T

                #st.write('image features size', image_features.shape)

                #st.write(f'PREDICTED SIZE : {model.predict(image_features)}')

                prediction = model.predict(image_features)[0]

                ftitle = " "*30
                ftitle += f"Predicted : {int(prediction)} | "
                ftitle += f"Real Size : {image_png_size} | "
                ftitle += f"Error : {((prediction / image_png_size - 1)*100):_.3f}%"

                
                fig = px.imshow(im_raw, title=ftitle)
                st.plotly_chart(fig, use_container_width=True)