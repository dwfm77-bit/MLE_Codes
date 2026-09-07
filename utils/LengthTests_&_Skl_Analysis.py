#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:20:28 2026

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
#from morph import mm
# import tifffile as tiff
import os, cv2
#from skimage.morphology import remove_small_objects
import tifffile as tiff
from skimage.morphology import reconstruction, skeletonize
#from skimage.filters import threshold_otsu
#import rasterio
#from rasterio.enums import Resampling
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import pandas as pd
# import seaborn as sns
from mm_functions import Length_Test
from joblib import load

###############################################################################


#==============================================================================
# Diretório das imagens
DS_DIR = '/home/usuario/Documentos/1.Doutorado/Datasets/S2-RD_lite/Test'
img_dir = os.path.join(DS_DIR, 'Imgs')
mask_dir = os.path.join(DS_DIR, 'Mask')

output_dir = '/home/usuario/Documentos/1.Doutorado/Testes/Testes_MLE/Ablation_Tests'
os.makedirs(output_dir, exist_ok=True)
# print_results = True

#=========================== REALCE LINEAR ====================================
band_list = ['ri']#['b2', 'b3', 'b4', 'b8', 'b11', 'b12', 'ndvi', 'ndbi', 'ndwi', 'ri']

dilation_list = ['b2','b3', 'b4', 'b11','b12','ndbi']
erosion_list = ['ri'] #['b8', 'ndvi', 'ndwi','ri']

gmm_mdl_path = "/home/usuario/Documentos/1.Doutorado/Testes/Testes_MLE/GMM_Model/GMM_tests_Bands__b2_b8_ri"

gmm_bands = ['b2', 'b8', 'ri']


B = np.ones((3,3),np.int8)

Length_list = list(range(3, 511, 2)) #(3, 511, 2)

thr_list = list(range(50, 100, 5))

NS = [512,512]
mm_operation = 'erosion' #'dilation'

#Ajusta o tamanho das fontes nos gráficos
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

all_metrics = [
    "iou", "f1", "precision", "recall",
    "skl_iou", "skl_f1", "skl_prec", "skl_rec", "tp", "fp", "fn", "tn"]


# results = {
#     metric: {b: [] for b in band_list}
#     for metric in all_metrics
# }

#Length tests
output_dir = '/home/usuario/Documentos/1.Doutorado/Datasets/temp'
length_out_dir = os.path.join(output_dir, 'Length_Tests')
os.makedirs(length_out_dir, exist_ok=True)


if mm_operation == 'dilation':

    # --------------------------------------------------
    # Experimentos
    # --------------------------------------------------
    for length in Length_list:
        print(f"- Comprimento B: {length}")
        for lp in thr_list:
            for band in band_list:
                print(f"    Banda: {band}")
    
                metrics = Length_Test(
                    DS_DIR, band, mm_operation, ste, length, skl_analysis=skl_a, LP=lp)
    
                for m in all_metrics:
                    results[m][band].append(metrics[m])
else:

    # --------------------------------------------------
    # Experimentos
    # --------------------------------------------------

    for band in band_list:
        print(f"    Banda: {band}")
        
        results = {
            LP: {metric: [] for metric in all_metrics}
            for LP in thr_list
        }

        for lp in thr_list:
            print(f"    LP: {lp}")

            for length in Length_list:
                print(f"- Comprimento B: {length}")
    
                metrics = Length_Test(
                    DS_DIR, band, mm_operation, B, length, skl_analysis=True, LP=lp)
    
                for m in all_metrics:
                    results[lp][m].append(metrics[m])
                
        
        for metric in all_metrics:
            
            # -----------------------------
            # Salva os dados em CSV
            # -----------------------------
            df = pd.DataFrame({
                "Length": Length_list
            })
            
            for lp in thr_list:
                df[f"LP_{lp}"] = results[lp][metric]
            
            df.to_csv(
                os.path.join(
                    length_out_dir,
                    f"{band}_{metric}_LP_en.csv"
                ),
                index=False
            )

            plt.figure(figsize=(10,6))
        
            for lp in thr_list:
        
                plt.plot(
                    Length_list,
                    results[lp][metric],
                    marker='o',
                    linewidth=2,
                    label=f'LP = {lp}%'
                )
                plt.xticks(Length_list[::10])
        
            plt.xlabel("Length of structuring element")
            plt.ylabel(metric.replace("_", " ").upper())
            plt.title(f"{metric.replace('_',' ').upper()} vs Structuring Element Length")
            plt.grid(True, alpha=0.3)
            plt.legend(
                title="Percentile threshold",
                ncol=2
            )
        
            plt.tight_layout()
        
            plt.savefig(os.path.join(length_out_dir, f"{band}_{metric}_LP_en.png"))
