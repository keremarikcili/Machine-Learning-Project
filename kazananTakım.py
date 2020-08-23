# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:33:29 2020

@author: kerem
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
 



dataset = pd.read_csv('high.csv')

X = dataset.iloc[:, [0]].values # Maç ID si

y = dataset.iloc[:, [1]].values # Mavi kazanan veya kırmızı kazanan

X= np.array(X)
y= np.array(y)


devam = "e"
kazanç=0
while(devam):
    
    gameID = int(input("Sizce hangi takım kazanır , Game ID giriniz :"))
    winner = int(input("Mavi takım mı ? Yoksa kırmızı takım mı? :(MAVİ TAKIM KAZANIR:1,KIRMIZI TAKOM KAZANIR:0)" ))
    
    gameIDFind = np.where(X==gameID)# game ID sini aldık
    
    gameWinnerFind = y[gameIDFind[0][0]] # takım tahminini aldık
    #datasetFind = np.where(y==gameIDFind)
    # oyunun kazananı ile ID sini karşılaştıracağız
    
    print(gameIDFind)
    deger = gameIDFind[0][0]
    print(gameWinnerFind[0])
    
    if(gameWinnerFind[0] == winner):
        kazanç = kazanç+50
        print("Toplam kazancınız :     ",kazanç)
    
    else:
        print("Tahmin edemediniz toplam kazancınız ", kazanç)
    
    
    deneyim=input("Tekrardan para tahmini yapmak ister misiniz ? ")  
    
    if(deneyim!="e"):
        break;
        
        
        
        
        
        
        
        
        
        
        