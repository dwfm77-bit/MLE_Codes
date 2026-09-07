#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:05:00 2025

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.mm_functions import Morphological_Linear_Features_Enhance_S2_imgs, plot_metrics_all_bands, metrics_dict_to_dataframe, extract_metric
from joblib import load
from pathlib import Path

###############################################################################
# Diretório raiz do projeto
PROJECT_DIR = Path(__file__).resolve().parent

# Dataset
#DS_DIR = Path("/home/usuario/Documentos/1.Doutorado/Datasets/S2-RD_lite/Test")
DS_DIR = PROJECT_DIR.parent / "Dataset" / "Test"
img_dir = DS_DIR / "Imgs"
mask_dir = DS_DIR / "Mask"

output_dir = PROJECT_DIR / "Results" / "MLE"
#output_dir = '/home/usuario/Documentos/1.Doutorado/Testes/Testes_MLE/Ablation_Tests'
os.makedirs(output_dir, exist_ok=True)

#=========================== Parametros ====================================
band_list = ['ri']#['b2', 'b3', 'b4', 'b8', 'b11', 'b12', 'ndvi', 'ndbi', 'ndwi', 'ri']

dilation_list = ['b2','b3', 'b4', 'b11','b12','ndbi']
erosion_list = ['b8', 'ndvi', 'ndwi','ri']

gmm_mdl_path = PROJECT_DIR / "GMM_Model" / "GMM_tests_Bands__b2_b8_ri"

gmm_bands = ['b2', 'b8', 'ri']

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


output_dir = os.path.join(output_dir, "Linear_Enhancement")
os.makedirs(output_dir, exist_ok=True)

metrics_dilation = {}
metrics_erosion  = {}

Length_list = list(range(15, 103, 2))
length_b = [35]


#abre o modelo gmm para filtragem
bundle = load(os.path.join(gmm_mdl_path,"gmm_model_best.joblib"))
gmm_model = bundle["model"]

print("Análise de realce linear\n")
    
for b in band_list:
    print(f"Analisando banda {b}")
    #os.makedirs(os.path.join(output_dir, b), exist_ok=True)

    if b in dilation_list:
        # -------------------------
        # Dilatação
        # -------------------------
        mm_operator = mm_operation[1]

        metrics = Morphological_Linear_Features_Enhance_S2_imgs(
            img_dir=DS_DIR,
            output_dir=output_dir,
            band=b,
            operacao=mm_operator,
            length_list = Length_list, #linear_b_length,
            NS=NS,
            ste=B,
            thickness=1,
            p_processing=True,
            channel_gmm=gmm_bands,
            skl_analysis=True,
            gmm_analysis=True,
            gmm_mdl=gmm_model,
            conditional_dil=True,
            save_results=False,
            show_imgs=False
        )


        metrics_dilation[b] = metrics

    else:
        # -------------------------
        # Erosão
        # -------------------------
        mm_operator = mm_operation[0]

        metrics = Morphological_Linear_Features_Enhance_S2_imgs(
            img_dir=DS_DIR,
            output_dir=output_dir,
            band=b,
            operacao=mm_operator,
            length_list = length_b, #linear_b_length,
            NS=NS,
            ste=B,
            thickness=1,
            p_processing=False,
            lp=95,
            channel_gmm=gmm_bands,
            skl_analysis=True,
            gmm_analysis=True,
            gmm_mdl=gmm_model,
            conditional_dil=True,
            save_results=False,
            show_imgs=True
        )

        metrics_erosion[b] = metrics
        

# =============== Erosion Boxplots ==================

plot_metrics_all_bands(
    metrics_dict={b: metrics_erosion[b]["iou"] for b in metrics_erosion},
    metric_name="IoU",
    title="IoU – Linear Enhancement by Erosion",
    ylabel="IoU"
)

plot_metrics_all_bands(
    metrics_dict={b: metrics_erosion[b]["f1"] for b in metrics_erosion},
    metric_name="F1-Score",
    title="F1-Score – Linear Enhancement by Erosion",
    ylabel="F1-Score"
)

plot_metrics_all_bands(
    metrics_dict={b: metrics_erosion[b]["skl_iou"] for b in metrics_erosion},
    metric_name="SKL-IoU",
    title="Skl-IoU – Linear Enhancement by Erosion",
    ylabel="Skl IoU"
)

plot_metrics_all_bands(
    metrics_dict={b: metrics_erosion[b]["skl_f1"] for b in metrics_erosion},
    metric_name="SKL-F1-Score",
    title="Skl-F1-Score – Linear Enhancement by Erosion",
    ylabel="Skl F1"
)

# =============== Dilation Boxplots ==================

plot_metrics_all_bands(
    metrics_dict={b: metrics_dilation[b]["iou"] for b in metrics_dilation},
    metric_name="IoU",
    title="IoU – Linear Enhancement by Dilation",
    ylabel="IoU"
)

plot_metrics_all_bands(
    metrics_dict={b: metrics_dilation[b]["f1"] for b in metrics_dilation},
    metric_name="F1-Score",
    title="F1-Score – Linear Enhancement by Dilation",
    ylabel="F1-Score"
)

plot_metrics_all_bands(
    metrics_dict={b: metrics_dilation[b]["skl_iou"] for b in metrics_dilation},
    metric_name="SKL-IoU",
    title="Skl-IoU – Linear Enhancement by Dilation",
    ylabel="Skl IoU"
)

plot_metrics_all_bands(
    metrics_dict={b: metrics_dilation[b]["skl_f1"] for b in metrics_dilation},
    metric_name="SKL-F1-Score",
    title="Skl-F1-Score – Linear Enhancement by Dilation",
    ylabel="Skl F1"
)
    
extract_metric(metrics_dilation, "iou")
extract_metric(metrics_dilation, "precision")
extract_metric(metrics_dilation, "recall")
extract_metric(metrics_dilation, "f1")
extract_metric(metrics_dilation, "skl_iou")
extract_metric(metrics_dilation, "skl_f1")
extract_metric(metrics_dilation, "bf_score")
extract_metric(metrics_dilation, "median_length_skl")

df_dilation = metrics_dict_to_dataframe(metrics_dilation, "dilation")
df_erosion  = metrics_dict_to_dataframe(metrics_erosion, "erosion")

df_dilation.to_csv(
os.path.join(output_dir, "metrics_linear_enhancement_dilation.csv"),
index=False
)

df_erosion.to_csv(
    os.path.join(output_dir, "metrics_linear_enhancement_erosion.csv"),
    index=False
)