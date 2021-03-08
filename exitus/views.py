import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64

from django.http import HttpResponse
from django.shortcuts import render, redirect
import requests
from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# importation de mon modèle
file_path = "models/lr_model.pkl"    
lr_model = pickle.load(open(file_path, 'rb'))
df_all = None

def submit(url):
    # Il s'agit de notre IA en question.
    # fonction recvant en paramètre le chemin d'accès à un fichier excel et 
    # retournant le resultat de son traitement dans un tableau de 0 et de 1.

    df_all = df = pd.read_csv(url)
    df_all = df_all.drop(["Exited"], axis = 1)
    #print(df_all["Surname"])
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
    x = df.iloc[:,0:10].values
    y = df.iloc[:, 10:11].values

    enc = OneHotEncoder(sparse=False)
    x1 = enc.fit_transform(x[:, 1].reshape(-1, 1))
    x = np.delete(x, 1, axis=1)
    x2 = enc.fit_transform(x[:, 1].reshape(-1, 1))
    x = np.delete(x, 1, axis=1)

    x = np.concatenate((x, x1, x2), axis=1)

    norm = RobustScaler()
    x = norm.fit_transform(x)
    y = norm.fit_transform(y.reshape(-1, 1))

    df_exited = pd.DataFrame({'Exited': lr_model.predict(x)}, index=range(0, len(lr_model.predict(x))))
    df_all = pd.concat([df_all, df_exited], axis=1, join="inner")
    #aff(df_all)
    return df_all
    
def home(request):
    return render(request, 'home.html')

def graphe(request):
    if request.GET['url']:
        url = request.GET['url']
        lr = submit(url)
        #aff(lr)
        """
        for r in lr["CustomerId"]:
            aff(r)
        """
        aff(lr["CustomerId"].iloc[:].values)
        return render(request, 'graphe.html', { 
            "subplot": g_subplot(lr), 
            "histplot": g_hist(lr), 
            "data": lr,
        })
    
    return render(request, 'home.html')

def receive_excel(request):
    if request.method == 'POST' and request.FILES['excel_file']:
        excel_file = request.FILES['excel_file']
        fs = FileSystemStorage()
        filename = fs.save(excel_file.name, excel_file)
        url = fs.url(filename)
        return redirect('/dashboard?url=http://127.0.0.1:8000{}'.format(url))
    
    return redirect('/home')

def load(request, url):
    plt.plot(range(10))
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return render(request, 'for_oh_for.html', { 'data': uri})


def g_subplot(df):
    # Reçoit en paramètre une dataframe
    # Dessine le pie-chart des clients retenus par la banque et ceux qui s'en sont allés
    # et retourne son image en base64

    labels = 'Clients partis de la banque', 'Clients Retenus par la banque'
    sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
    #sizes = [len(y[y==1]), len(y[y==0])]
    explode = (0, 0.1)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Proportion des clients qui ont pu être retenu \net ceux qui ont quitté la banque", size = 20)
    #plt.show()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri

def g_hist(df):
    # histogramme des variables catégorielles
    fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
    sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
    sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
    sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
    sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])

    #sns.set()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri
"""
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")
f, ax = plt.subplots(figsize=(9, 6))
new = sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

figure = io.BytesIO()
new.get_figure().savefig(figure, format='png')
image_file = ImageFile(figure.getvalue())
"""

def aff(v):
    print('\n\n{}\n'.format(v))