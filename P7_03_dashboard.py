# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:26:00 2020

@author: Matthieu
"""


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import lightgbm
import shap
import streamlit.components.v1 as components
import joblib

### récupérer les données 
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
y_train = y_train.drop(['Unnamed: 0'],axis=1)
X_train = X_train.drop(['Unnamed: 0'],axis=1)
prédiction = pd.read_csv('prédiction.csv')
model = joblib.load('lgb.pkl')

###

st.title('Dashboard Scoring Credit')

clients_nom = st.sidebar.selectbox(
    'Veuillez saisir l\'identifiant d\'un client:',
      prédiction)

prédiction = prédiction*100



### prédiction

st.write('le client à :',round(prédiction['1'],2).loc[clients_nom], '% de chance de ne pas rembouser')


## graphique centrale

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
       
shap.initjs()
explainer = shap.TreeExplainer(model,feature_dependence="independent")
shap_values = explainer.shap_values(X_train)

st_shap(shap.force_plot(explainer.expected_value[1], 
                        shap_values[1][clients_nom,:],
                        X_train.iloc[clients_nom,:],link="logit"))



###### graphique en dessous 

df = pd.concat([X_train,y_train],axis=1)

moy = ['moy']
risque=['ris']
pas_risque = ['no_ris']
indiv = ['indiv']


fig =  make_subplots(rows=1, 
                      cols=3,
                      subplot_titles=("ext_source_3",
                                      "ext_source_2",
                                      "days birth"))

fig.add_trace(
    go.Bar(x=moy,y=[df['EXT_SOURCE_3'].mean()]),row=1, col=1)
fig.add_trace(
    go.Bar(x=risque,y=[df['EXT_SOURCE_3'].loc[df['TARGET']==1].mean()]),
    row=1, col=1)
fig.add_trace(
    go.Bar(x=pas_risque,y=[df['EXT_SOURCE_3'].loc[df['TARGET']==0].mean()]),
    row=1, col=1)
fig.add_trace(
    go.Bar(x=indiv,y=[df['EXT_SOURCE_3'].loc[clients_nom]]),
    row=1, col=1)

fig.add_trace(
    go.Bar(x=moy,y=[df['EXT_SOURCE_2'].mean()]),row=1, col=2)
fig.add_trace(
    go.Bar(x=risque,y=[df['EXT_SOURCE_2'].loc[df['TARGET']==1].mean()]),
    row=1, col=2)
fig.add_trace(
    go.Bar(x=pas_risque,y=[df['EXT_SOURCE_2'].loc[df['TARGET']==0].mean()]),
    row=1, col=2)
fig.add_trace(
    go.Bar(x=indiv,y=[df['EXT_SOURCE_2'].loc[clients_nom]]),
    row=1, col=2)

fig.add_trace(
    go.Bar(x=moy,y=[df['DAYS_BIRTH'].mean()]),row=1, col=3)
fig.add_trace(
    go.Bar(x=risque,y=[df['DAYS_BIRTH'].loc[df['TARGET']==1].mean()]),
    row=1, col=3)
fig.add_trace(
    go.Bar(x=pas_risque,y=[df['DAYS_BIRTH'].loc[df['TARGET']==0].mean()]),
    row=1, col=3)
fig.add_trace(
    go.Bar(x=indiv,y=[df['DAYS_BIRTH'].loc[clients_nom]]),
    row=1, col=3)

fig.update_layout(height=400, width=800)
st.plotly_chart(fig)

st.write('moy = la moyenne de la feature')
st.write('ris = la moyenne des clients defaut de paiement')
st.write('no_ris = la moyenne des clients régulier')
st.write('indiv = le score du client concerné')

### graphique coté (meilleurs feature)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.pyplot(shap.summary_plot(shap_values[1],X_train))


st.write('Ext_source = Score normalisé à partir d\'une source de données externe.\
          Une sorte de cote de crédit cumulative établie à l\'aide de nombreuses sources de données')
st.sidebar.write('code gender = Homme ou Femme')
st.sidebar.write('days_birth = date de naissance')
st.sidebar.write('days last phone change = la date de changement du dernier téléphone')
st.sidebar.write('days employed = depuis combien de temps le client travail')











