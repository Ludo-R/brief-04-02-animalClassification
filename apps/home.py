#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:06:37 2020

@author: randon
"""

import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app, server
import datetime
from skimage.transform import resize
import pickle
from transformm import RGB2GrayTransformer, HogTransformer
from PIL import Image
import base64
from io import BytesIO
import numpy as np

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the Animal Search API", className="text-center")
                    , className="mb-4 mt-4")
        ]),
        dbc.Row([
            dbc.Col(html.Img(src="/assets/home2.jpg", height="300px")
                    , className="mb-4 text-center")
            ]),
        dbc.Row([
            dbc.Col(html.H4(children='Animal Image Detector'
                                     ))
            ]),
        dbc.Row([
            dbc.Col(html.H5(children='Here you can Upload any image to dectect label, you can detect animals below :')                        
                    , className="mb-4")
            ]),
        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H5(children='Chicken',
                                               className="text-center"),
                                       html.Img(src="/assets/chicken.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Monkey',
                                               className="text-center"),
                                       html.Img(src="/assets/monkey.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Bear',
                                               className="text-center"),
                                       html.Img(src="/assets/bear.jpg", height="70px")]),),
                        dbc.Col(dbc.Card(children=[html.H5(children='Pandas',
                                               className="text-center"),
                                       html.Img(src="/assets/panda.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Deer',
                                               className="text-center"),
                                       html.Img(src="/assets/deer.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Eagle',
                                               className="text-center"),
                                       html.Img(src="/assets/eagle.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Elephant',
                                               className="text-center"),
                                       html.Img(src="/assets/elephant.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Spider',
                                               className="text-center"),
                                       html.Img(src="/assets/spider.jpg", height="70px")]),className="mb-4"),
            ]),
        dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'background-color':'gray'
        },
        # Allow multiple files to be uploaded
        multiple=False
        ),
        html.Div(id='output-image-upload', className="mb-4"),
        dbc.Row([
            dbc.Col(html.H5(children='Animal Image Searcher')
                    , className="mb-4"),]),
        dbc.Row([
            dbc.Col(html.Img(src="/assets/pates.jpg", height="150px")
                    , className="mb-4 text-center"),
            dbc.Col(dbc.Card(children=[html.H3(children='Google Image like Exclusive for Animals',
                                               className="text-center"),
                                       dbc.Button("Go on Animal Engine", href="/page1",
                                                                   color="primary",
                                                        className="mt-3"),                              
                                       ],
                             body=True, color="dark", outline=True,className="align-self-center")
                    , width=8, className="mb-4 text-center"),
            dbc.Col(html.Img(src="/assets/pates.jpg", height="150px")
                    , className="mb-4 text-center")
        ]),
        html.A("Get the full code of app on my github repositary",
               href="https://github.com/Ludo-R/")
])])
def load_and_preprocess(image):
    encoded_image = image.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    image = Image.open(bytes_image).convert('RGB')
    impred = image.resize((80,80))
    return impred

def np_array_normalise(test_image):
   np_image = np.array(test_image)
   final_image = np.expand_dims(np_image, 0)
   return final_image

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),)

def prediction(image):
    if image is not None:
        final_img = load_and_preprocess(image)
        final_img = np_array_normalise(final_img)
        with open('/Users/buu/devia/brief-04-02-animalClassification/hog_sgd_model.pkl', 'rb') as f1:
            pred = pickle.load(f1)
        impred = pred.predict(final_img)
        return html.Div([
            html.H1(impred) ,
            html.Img(src=image, style={'height':'30%', 'width':'30%'}),
            html.Hr(),
            ], className="text-center")

    