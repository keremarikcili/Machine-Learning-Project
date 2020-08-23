# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:49:20 2020

@author: kerem
"""
"""
FİNAL PROJESİ uygulaması DecisionTreeClassifier Ve KNeighborsClassifier kullanılarak sınıflandırma yapılmıştır

OZL16000222 - KEREM ARIKÇILI
05160000303 - EMRE DURSUN



"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate 
 
 
from sklearn.metrics import accuracy_score
from numpy import mean 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression as lr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# Importing the dataset
dataset = pd.read_csv('high.csv')
 
X = dataset.iloc[:, [5,6,7]].values
z = dataset.iloc[:,[5]].values
t = dataset.iloc[:,[6]].values 
w = dataset.iloc[:,[7]].values 
y = dataset.iloc[:, [12]].values

X= np.array(X)
y= np.array(y)
"""
say=0
for col in dataset.columns:
    print(say , "" ,col)
    say+=1
"""


# Dataset eğitim ve test olmak üzere 2 parçaya ayrıldı.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,test_size = 0.3, random_state = 0)
 
  
 
clf = DecisionTreeClassifier(random_state=0)
# Verinin %70'ini Eğitim, %30'unu test verisi olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 0)
# Eğitim Verisi ile eğitimi gerçekleştiriyoruz
clf.fit(X_train,y_train)
test_sonuc_Blue = clf.predict(X_test)
 
cm_Blue = confusion_matrix(y_test, test_sonuc_Blue)
 




  

 
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(clf, X = X_train, y = y_train, cv = 5)
print("K-Fold Cross :",accuracies)
print("mean :",accuracies.mean())

print("standart sapma :",accuracies.std())

plt.plot(X,y)
plt.scatter(z,y,s=75,c="red")
plt.scatter(t,y,s=75,c="purple")
plt.scatter(w,y,s=75,c="brown")
plt.xlabel("Blues Kills/Assists/Deaths sayısı")
plt.ylabel("Para")

plt.legend(('Kills', 'death','assist'), loc=4, scatterpoints=1)


plt.show()

Z = dataset.iloc[:,[24,25,26]]
q = dataset.iloc[:,[24]].values#Red kills
e = dataset.iloc[:,[31]].values#Red totalMoney
r = dataset.iloc[:,[25]].values#Red deaths
o = dataset.iloc[:,[26]].values#Red Assists

Z= np.array(Z)
e= np.array(e)

plt.plot(Z,e)
# Dataset eğitim ve test olmak üzere 2 parçaya ayrıldı.
from sklearn.model_selection import train_test_split
X_train_Red, X_test_Red, y_train_Red, y_test_Red = train_test_split(Z, e, train_size=0.7,test_size = 0.3, random_state = 0)
 

#Redler için  decision Tree
 
clf = DecisionTreeClassifier(random_state=0)
# Verinin %70'ini Eğitim, %30'unu test verisi olarak ayırıyoruz
X_train_Red, X_test_Red, y_train_Red, y_test_Red = train_test_split(Z,e, train_size = 0.7, test_size = 0.3, random_state = 0)
# Eğitim Verisi ile eğitimi gerçekleştiriyoruz
clf.fit(X_train_Red,y_train_Red)
test_sonuc_Red = clf.predict(X_test_Red)
 
cm_Red = confusion_matrix(y_test_Red, test_sonuc_Red)


#K-fold red

from sklearn.model_selection import cross_val_score
accuracies_Red = cross_val_score(clf, X = X_train_Red, y = y_train_Red, cv = 5)
print("K-Fold Cross :",accuracies_Red)
print("mean :",accuracies_Red.mean())

#KIRMIZI SHOW()
plt.plot(Z,e)
plt.scatter(q,e,s=75,c="red")
plt.scatter(r,e,s=75,c="purple")
plt.scatter(o,e,s=75,c="brown")
plt.xlabel("Reds Kills/Assists/Deaths sayısı")
plt.ylabel("Para")

plt.legend(('Kills', 'death','assist'), loc=4, scatterpoints=1)


plt.show()

x = dataset.iloc[:, [5]].values
y = dataset.iloc[:, [12]].values
z = dataset.iloc[:, [6]].values
t = dataset.iloc[:, [7]].values
 
x=np.array(x)
y=np.array(y)
z=np.array(z)
t=np.array(t)

# KIRMIZI TAKIM


#MAVİ TAKIM reShape
x = np.reshape(x,9879)#X ekseninde yer alan eleman sayısı kadar array oluşturur.
y = np.reshape(y,9879)#Y ekseninde yer alan eleman sayısı kadar array oluşturur.
z = np.reshape(z,9879) 
t = np.reshape(t,9879)
#KIRMIZI TAKIM reShape
q = np.reshape(q,9879)#X ekseninde yer alan eleman sayısı kadar array oluşturur.
e = np.reshape(e,9879)#Y ekseninde yer alan eleman sayısı kadar array oluşturur.
r = np.reshape(r,9879) 
o = np.reshape(o,9879)
 
lineer = lr() # Lineer Regresyonu çağırdık.

 
y=y.reshape(-1,1) #(-1,1) yazmamızın sebebi numpy kütüphanesi
                  # grafiğimizin boyutlarını arrayimizin boyutuna göre ayarlamasıdır.
x=x.reshape(-1,1)  

z=z.reshape(-1,1) 
t=t.reshape(-1,1)

q=q.reshape(-1,1) #(-1,1) yazmamızın sebebi numpy kütüphanesi
                  # grafiğimizin boyutlarını arrayimizin boyutuna göre ayarlamasıdır.
e=e.reshape(-1,1)  

r=r.reshape(-1,1) 
o=o.reshape(-1,1)



lineer.fit(x,y)  
m = lineer.coef_ # Eğrimizin eğimini bulduk.  

b= lineer.intercept_ # Eğim denklemindeki +b dir.

a = np.arange(30) 

a=a.reshape(-1,1)  
lineer.fit(z,y)

z_m = lineer.coef_ # Eğrimizin eğimini bulduk.  

z_b= lineer.intercept_ # Eğim denklemindeki +b dir.

z_a = np.arange(30) 

z_a=z_a.reshape(-1,1)  

lineer.fit(t,y)
lineer.predict(y) 

y_m = lineer.coef_ # Eğrimizin eğimini bulduk.  

y_b= lineer.intercept_ # Eğim denklemindeki +b dir.

y_a = np.arange(30) 

y_a=y_a.reshape(-1,1)  

#KIRMIZI TAKIM 

lineer.fit(q,y)  #kill
q_m = lineer.coef_ # Eğrimizin eğimini bulduk.  

q_b= lineer.intercept_ # Eğim denklemindeki +b dir.

q_a = np.arange(30) 

q_a=q_a.reshape(-1,1)  

lineer.fit(r,y) # deaths

r_m = lineer.coef_ # Eğrimizin eğimini bulduk.  

r_b= lineer.intercept_ # Eğim denklemindeki +b dir.

r_a = np.arange(30) 

r_a=r_a.reshape(-1,1)  

lineer.fit(o,y) # assists
lineer.predict(y) 

o_m = lineer.coef_ # Eğrimizin eğimini bulduk.  

o_b= lineer.intercept_ # Eğim denklemindeki +b dir.

o_a = np.arange(30) 

o_a=o_a.reshape(-1,1)  

# KNN Algoritması ile Mavi Takım Analizi
X_train_BK, X_test_BK, y_train_BK, y_test_BK = train_test_split(X, y, test_size = 0.55, random_state = 0)


sc = StandardScaler()
X_train_BK = sc.fit_transform(X_train_BK)
X_test_BK = sc.transform(X_test_BK)

classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train_BK, y_train_BK)

y_pred_BK = classifier.predict(X_test_BK)



# KNN Algoritması ile Kırmızı Takım Analizi

X_train_RK, X_test_RK, y_train_RK, y_test_RK = train_test_split(Z, e, test_size = 0.55, random_state = 0)


sc = StandardScaler()
X_train_RK = sc.fit_transform(X_train_RK)
X_test_RK = sc.transform(X_test_RK)

classifier_RK = KNeighborsClassifier(n_neighbors = 2)
classifier_RK.fit(X_train_RK, y_train_RK)

y_pred_RK = classifier.predict(X_test_RK)



print(cm_Red,"burası kırmızı decion treeler")
print(cm_Blue,"burası mavi decion treeler")

#Knn Sonucu Ortalamaları
 
print("Blue Team KNN SONUCU ORTALAMASI : ",round(mean(y_pred_BK),2))
print("Red Team KNN SONUCU ORTALAMASI  : " ,round(mean(y_pred_RK),2))
print("Blue Team DECISION TREE SONUCU ORTALAMASI : ",round(mean(test_sonuc_Blue),2))
print("Red Team DECISION TREE SONUCU ORTALAMASI  : " ,round(mean(test_sonuc_Red),2))




deneyim="e"
   
while(deneyim=="e"):
    
    kills = float(input("Oyun içerisinde kaç kill aldınız ?")) #
    assist = float(input("Oyun içerisinde kaç assist aldınız ?"))
    death = float(input("Oyun içerisinde kaç defa öldünüz ?"))
    
    tahmin_kills = m*kills+b  
    tahmin_assist = z_m*assist+z_b
    tahmin_death = y_m*death+y_b
    
    tahmin_kills_red = q_m*kills+q_b
    tahmin_assist_red = r_m*assist+r_b
    tahmin_death_red = o_m*death+o_b 
    
    
    plt.scatter(kills,tahmin_kills,c="black",marker="|") #noktamızın şeklini belirtiyoruz
    plt.xlabel("Kill sayısı")
    plt.ylabel("Para")
     
    print("Blue gamer kills TotalMoney = ",tahmin_kills)
    plt.plot(a,m*a+b) #eğim doğrusunu çizdirmemiz için gerekli
    
      #grafiğimizi gösteriyoruz
    
    plt.scatter(assist,tahmin_assist,c="black",marker=">")
    plt.xlabel("assist sayısı")
    plt.ylabel("Para")
    
    plt.plot(z_a,z_m*z_a+z_b)
    
    print("Blue gamer assist TotalMoney = ",tahmin_assist)
    plt.scatter(death,tahmin_death,c="black",marker="<")
    plt.xlabel("Kill/assist/death sayısı")
    plt.ylabel("Para")
    plt.title("Lol gaming TotalMoney tahmini ")
    plt.plot(y_a,y_m*y_a+y_b)
    print("Blue gamer death TotalMoney = ",tahmin_death)
   
    print("Blue gamer average TotalMoney ",(tahmin_kills+tahmin_assist+tahmin_death)/3)
    
    plt.legend(('Kills', 'death','assist'), loc=2, scatterpoints=1)
    
    
    plt.show()
    #KIRMIZI KİLLS DEATHS ASSISTS SHOW()
    
    plt.scatter(kills,tahmin_kills_red,c="black",marker="|") #noktamızın şeklini belirtiyoruz
    plt.xlabel("Kill sayısı")
    plt.ylabel("Para")
     
    print("Red gamer kills TotalMoney = ",tahmin_kills_red)
    plt.plot(tahmin_kills_red) #eğim doğrusunu çizdirmemiz için gerekli
    
      #grafiğimizi gösteriyoruz
    
    plt.scatter(assist,tahmin_assist,c="black",marker=">")
    plt.xlabel("assist sayısı")
    plt.ylabel("Para")
    
    plt.plot(r_a,r_m*r_a+r_b)
    
    print("Red gamer assist TotalMoney = ",tahmin_assist_red)
    plt.scatter(death,tahmin_death_red,c="black",marker="<")
    plt.xlabel("Kill/assist/death sayısı")
    plt.ylabel("Para")
    plt.title("Lol gaming TotalMoney tahmini ")
    plt.plot(o_a,o_m*o_a+o_b)
    print("Red gamer death TotalMoney = ",tahmin_death_red)
   
    print("Red gamer average TotalMoney ",(tahmin_kills_red+tahmin_assist_red+tahmin_death_red)/3)
    
    plt.legend(('Kills', 'death','assist'), loc=3, scatterpoints=1)
    plt.show()
    
    
    
   
    deneyim=input("Tekrardan totalMoney tahmini yapmak ister misiniz ? ")  
            
    
    if(deneyim!="e"):
        break        
    
    





























"""

OZL16000222 - KEREM ARIKÇILI

"""