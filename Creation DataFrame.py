# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:13:24 2021

@author: Olivier
"""


import os
from PIL import Image


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt 




class DataExtraction:

    def __init__(self):
        
        self.path = r"C:\Users\Olivier\Desktop\DATASCIENTEST\chest_xray\DATA"    

        self.df = pd.DataFrame(columns = ['fichier', 'format', 'rep1', 'rep2', 'pathogen',
                             'height', 'width', 'size', 'mode', 'mean', 
                             'median', 'std', 'seuil100'])

        self.compteur = 0

    def load_data(self, display = False, compteur_max = 10_000):
        
        self.compteur = 0
    
        if (compteur_max > 10) and (display == True):
            display = False
    
        for root, dirs, files in os.walk(self.path + '\\'): 
            
            for i in files:
                
                self.compteur += 1
                
                # Afficher un état d'avancement
                print('.', end='')
                
                full_path = os.path.join(root, i)
        
                data = { }        
        
                img = Image.open(full_path)
                        
                # Nom du fichier
                head, tail = os.path.split(full_path)
               
                data['fichier'] = tail
                
                # Répertoire 'NORMAL' OU 'PNEUMONIA'
                head, tail = os.path.split(head)
        
                data['rep2'] = tail
        
                # Répertoire 'train', 'test' ou 'val'
                head, tail = os.path.split(head)
        
                data['rep1'] = tail
          
                # Dimension de l'image
                data['height'] = img.height
                
                data['width'] = img.width
        
                # Size: nombre total de pixels
                data['size'] = img.height * img.width
        
                # Format (JPEG)
                data['format'] = img.format
                
                # Mode : couleur ou N&B
                data['mode'] = img.mode
        
                # Virus ou Bactérie si Pneumonie
                data['pathogen'] = ''
                
                if 'virus' in data['fichier']:
                    data['pathogen'] = 'virus'
                elif 'bacteria' in data['fichier']:
                    data['pathogen'] = 'bacteria'
        
        
                # Conversion en array numpy            
                np_img = np.array(img)
                        
                # Statistiques
                data['mean'] = round(np.mean(np_img), 2)
                
                data['median'] = round(np.median(np_img), 2)
                
                data['std'] = round(np.std(np_img), 2)
        
                if display == True:
                    plt.figure(figsize=(4,4))
                    plt.title(data['fichier'])
                    plt.imshow(img, cmap='gray')
        
                # Calcul du seuil (valeur = 100)
        
                if data['mode'] == 'RGB':
                    np_img = np_img[:, :, 0]
               
                img_seuil = self.seuillage(np_img, 100)
               
                up_to_seuil = round(img_seuil.sum() / data['size'] * 100, 2)
                
                data['seuil100'] = up_to_seuil        
        
                if display == True:
                    plt.figure(figsize=(4,4))
                    plt.title(data['fichier'])
                    plt.imshow(img_seuil, cmap='gray')
              
       
                # Ajout de la ligne dans le DataFrame
                self.df = self.df.append(data, ignore_index=True)        
            
            
                # Mettre fin à la boucle
                if self.compteur >= compteur_max:
                    return True


    def save_csv(self, file):
        
        self.df.to_csv(file, sep=';', index = False)

    
    def info(self):
        print('Nombre de fichiers traités: ', self.compteur)


    def seuillage(self, image, seuil):
    
        resultat = image.copy()
   
        resultat = image > seuil
    
        return resultat
   


extract = DataExtraction()

# Extraire les données de 3 fichiers et afficher images correspondantes
extract.load_data(display = True, compteur_max = 3)

#extract.load_data()

extract.info()

#extract.save_csv('data.csv')




