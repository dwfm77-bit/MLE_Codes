#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:10:38 2026

@author: usuario
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

erosion_dir = "/home/usuario/Documentos/1.Doutorado/Testes/Testes_MLE/1. Length Tests/Length_Tests_LP0%/length_tests_erosion.xlsx"
dilation_dir = "/home/usuario/Documentos/1.Doutorado/Testes/Testes_MLE/1. Length Tests/Length_Tests_LP0%/length_tests_dilation.xlsx"

output_dir = "/home/usuario/Documentos/1.Doutorado/Datasets/temp"

operation = 'erosion' #'dilation' # 'erosion'

if operation == 'dilation':
    input_dir = dilation_dir
    band_list = ["b2","b3","b4","ndbi", "b11", "b12"]
    
elif operation == 'erosion':
    input_dir = erosion_dir
    band_list = ["b8","ndvi", "ndwi", "ri"]


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
    "skl_iou", "skl_f1", "skl_prec", "skl_rec",
    "tp", "fp", "fn", "tn"
]


results = {
    metric: {band: [] for band in band_list}
    for metric in all_metrics
}


df1 = pd.read_excel(input_dir, sheet_name=None)

for metric in all_metrics:
    df2 = df1[metric]

    # comprimento do elemento estruturante
    length_list = df2["Length"].tolist()

    if operation == 'dilation':
        # bandas da segunda aba
        for band in ["b2", "b3", "b4", "b11", "b12", "ndbi"]:
            results[metric][band] = df2[band].tolist()
    else:
        # bandas da primeira aba
        for band in ["b8", "ndvi", "ndwi", "ri"]:
            results[metric][band] = df2[band].tolist()
       

# --------------------------------------------------
# Gráficos
# --------------------------------------------------
#inglês
for metric in all_metrics:
    plt.figure(figsize=(10, 6))
    for band in band_list:
        plt.plot(
            length_list,
            results[metric][band],
            marker="o",
            label=f"Band {band}"
        )
 
    plt.xlabel("Length of structuring element")
    plt.ylabel(metric.replace("_", " ").upper())
    plt.title(f"{metric.replace('_', ' ').upper()} vs B length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
 
    plt.savefig(
        os.path.join(
            output_dir,
            f"{metric}_{operation}_en.png"
        ),
        dpi=300
    )
    plt.show()
 
#português
for metric in all_metrics:
    plt.figure(figsize=(10, 6))
    for band in band_list:
        plt.plot(
            length_list,
            results[metric][band],
            marker="o",
            label=f"Banda {band}"
        )
 
    plt.xlabel("Tamanho do Elemento Estruturante")
    plt.ylabel(metric.replace("_", " ").upper())
    plt.title(f"{metric.replace('_', ' ').upper()} X Tamanho de B")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
 
    plt.savefig(
        os.path.join(
            output_dir,
            f"{metric}_{operation}_pt.png"
        ),
        dpi=300
    )
    plt.show()
         