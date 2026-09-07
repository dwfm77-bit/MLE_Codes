#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:05:00 2025

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from utils.mm_functions import  summarize_length_results, Test_SklThr_process

###############################################################################

# Diretório raiz do projeto
PROJECT_DIR = Path(__file__).resolve().parent

# Dataset
DS_DIR = PROJECT_DIR.parent / "Dataset" / "Test"
img_dir = DS_DIR / "Imgs"
mask_dir = DS_DIR / "Mask"

# Diretório dos resultados
thr_out_dir = PROJECT_DIR / "Results" / "Skl_Threshold_Tests"
thr_out_dir.mkdir(parents=True, exist_ok=True)

#=========================== Parametros ====================================
band_list = ['ri']#['b2', 'b3', 'b4', 'b8', 'b11', 'b12', 'ndvi', 'ndbi', 'ndwi', 'ri']

dilation_list = ['b2','b3', 'b4', 'b11','b12','ndbi']
erosion_list = ['b8', 'ndvi', 'ndwi','ri']

B = np.ones((3,3),np.int8)

Length_list = list(range(3, 511, 2))

thr_list = list(range(50, 100, 5))

linear_b_length = 101 # Valor final ajustado de B == 65

NS = [512,512]
mm_operation = ['erosion', 'dilation']

#Ajusta o tamanho das fontes nos gráficos
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

#==============================================================================

Thr_Tests_Metrics_dilation = Test_SklThr_process(DS_DIR, thr_out_dir, linear_b_length, thr_list, dilation_list, 'dilation', B)


df_val_dil, df_len_dil = summarize_length_results(
Thr_Tests_Metrics_dilation,
Length_list,
dilation_list)

Thr_Tests_Metrics_erosion = Test_SklThr_process(DS_DIR, thr_out_dir, linear_b_length, thr_list, erosion_list, 'erosion', B)

df_val_ero, df_len_ero = summarize_length_results(
Thr_Tests_Metrics_erosion,
Length_list,
erosion_list)


out_csv_val = os.path.join(
    thr_out_dir, "thr_tests_best_metric_value_dilation.csv"
)
out_csv_len = os.path.join(
    thr_out_dir, "thr_tests_best_length_dilation.csv"
)

df_val_dil.to_csv(out_csv_val, sep=";", decimal=",", encoding="utf-8-sig")
df_len_dil.to_csv(out_csv_len, sep=";", decimal=",", encoding="utf-8-sig")


out_csv_val = os.path.join(
    thr_out_dir, "thr_tests_best_metric_value_erosion.csv"
)
out_csv_len = os.path.join(
    thr_out_dir, "thr_tests_best_length_erosion.csv"
)

df_val_ero.to_csv(out_csv_val, sep=";", decimal=",", encoding="utf-8-sig")
df_len_ero.to_csv(out_csv_len, sep=";", decimal=",", encoding="utf-8-sig")

#Salva os resultados finais do teste de dilatação em xlsx
history_dilation = os.path.join(thr_out_dir, "thr_tests_dilation.xlsx")
with pd.ExcelWriter(history_dilation, engine="openpyxl") as writer:

   for metric in Thr_Tests_Metrics_dilation.keys():

       df = pd.DataFrame(
           {"Threshold": thr_list}
       )

       for band in Thr_Tests_Metrics_dilation[metric]:
           df[str(band)] = Thr_Tests_Metrics_dilation[metric][band]

       df.to_excel(
           writer,
           sheet_name=metric[:31],  # limite do Excel
           index=False
       )

#Salva os resultados finais do teste de erosão em xlsx
history_erosion = os.path.join(thr_out_dir, "thr_tests_erosion.xlsx")
with pd.ExcelWriter(history_erosion, engine="openpyxl") as writer:

   for metric in Thr_Tests_Metrics_erosion.keys():

       df = pd.DataFrame(
           {"Threshold": thr_list}
       )

       for band in Thr_Tests_Metrics_erosion[metric]:
           df[str(band)] = Thr_Tests_Metrics_erosion[metric][band]

       df.to_excel(
           writer,
           sheet_name=metric[:31],  # limite do Excel
           index=False
       )    