#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:20:28 2026

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.mixture import GaussianMixture
from morph import mm
# import tifffile as tiff
import os, cv2
#from skimage.morphology import remove_small_objects
# import tifffile as tiff
# from skimage.morphology import reconstruction, skeletonize
#from skimage.filters import threshold_otsu
#import rasterio
#from rasterio.enums import Resampling
from PIL import Image
# from scipy.ndimage import distance_transform_edt
# from tqdm import tqdm
import pandas as pd
# import seaborn as sns
from mm_functions import Length_Test, linear_enhance, skeleton_analysis, Calc_Metrics
import tifffile as tiff
from tqdm import tqdm

# from joblib import load

############################ FUNCTIONS #####################################
def MLE_function(ds_dir, band, operacao, length, output_path=None, save_img=False):

    image_files = sorted([
        f for f in os.listdir(os.path.join(ds_dir, band))
        if f.lower().endswith((".tif", ".tiff"))
    ])

    # Lista para armazenar as imagens realçadas
    enhanced_images = []
    image_names = []
    
    if save_img:
        os.makedirs(output_path, exist_ok=True)

    for f in image_files:

        # --- Leitura da banda ---
        img = tiff.imread(
            os.path.join(ds_dir, band, f)
        )

        if img.ndim > 2:
            img = img[:, :, 0]

        # --- Realce linear ---
        img_enhanced = linear_enhance(img, length, operacao)

        # --- Armazenamento na lista ---
        enhanced_images.append(img_enhanced)
        image_names.append(f)

        # --- Salvamento da imagem realçada ---
        if save_img:
            tiff.imwrite(
                os.path.join(output_path, f),
                img_enhanced
            )

    return enhanced_images, image_names

#==============================================================================
# Diretório das imagens
DS_DIR = '/home/usuario/Documentos/1.Doutorado/Datasets/S2-RD/Test'
img_dir = os.path.join(DS_DIR, 'Imgs')
mask_dir = os.path.join(DS_DIR, 'Mask')

output_dir = '/home/usuario/Documentos/1.Doutorado/Datasets/temp/Length_Tests-x-Skl_Analysis'
os.makedirs(output_dir, exist_ok=True)
# print_results = True

#=========================== REALCE LINEAR ====================================
band_list = ['ri']#['b2', 'b3', 'b4', 'b8', 'b11', 'b12', 'ndvi', 'ndbi', 'ndwi', 'ri']

dilation_list = ['b2','b3', 'b4', 'b11','b12','ndbi']
erosion_list = ['ri'] #['b8', 'ndvi', 'ndwi','ri']

# gmm_mdl_path = "/home/usuario/Documentos/1.Doutorado/Testes/Testes_MLE/GMM_Model/GMM_tests_Bands__b2_b8_ri"

# gmm_bands = ['b2', 'b8', 'ri']


B = np.ones((3,3),np.int8)

Length_list = list(range(3, 513, 2)) #(3, 513, 2)

thr_list = list(range(50, 100, 5))

NS = [512,512]
mm_operation = 'erosion'
mm_operation = 'plot_graph'

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


results = {
    metric: {b: [] for b in band_list}
    for metric in all_metrics
}

#Length tests
output_dir = '/home/usuario/Documentos/1.Doutorado/Datasets/temp'
length_out_dir = os.path.join(output_dir, 'Skl_Thr_X_Length_Tests_RI')
os.makedirs(length_out_dir, exist_ok=True)


# Métricas calculadas diretamente por Calc_Metrics
base_metrics = [
    "iou", "f1", "precision", "recall",
    "tp", "fp", "fn", "tn"
]

# Experimentos de erosão
if mm_operation == "erosion":

    for band in band_list:

        print(f"\nBanda: {band}")

        # results[LP][métrica] = lista de médias, uma por comprimento
        results = {
            lp: {metric: [] for metric in base_metrics}
            for lp in thr_list
        }

        for length in Length_list:

            #print(f"Comprimento B: {length}")

            # Realça todas as imagens da banda para este comprimento
            enhanced_images, image_names = MLE_function(
                img_dir, band, mm_operation, length
            )

            # Processa cada percentil
            #for lp in thr_list:
            for lp in tqdm(thr_list, desc=f"Processando os percentis para o Comprimento {length}"):

                #print(f"    LP: {lp}%")

                metrics = {metric: [] for metric in base_metrics}

                for idx, img_enhanced in enumerate(enhanced_images):

                    # Análise morfológica e filtragem
                    img_pp, _ = skeleton_analysis(
                        img_enhanced,
                        ste=B,
                        dilation=True,
                        seg_filter=True,
                        length_percentile=lp
                    )

                    # Máscara de referência
                    f = image_names[idx]

                    mask_ref = (
                        np.array(
                            Image.open(
                                os.path.join(mask_dir, f)
                            ).convert("L")
                        ) > 0
                    )

                    # Verifica a compatibilidade espacial
                    if img_pp.shape != mask_ref.shape:
                        raise ValueError(
                            f"Dimensões incompatíveis em {f}: "
                            f"processada={img_pp.shape}, "
                            f"máscara={mask_ref.shape}"
                        )

                    # Calcula as métricas
                    (
                        iou, precision, recall, f1,
                        tp, fp, fn, tn
                    ) = Calc_Metrics(img_pp, mask_ref)

                    values = {
                        "iou": iou,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn
                    }

                    for metric, value in values.items():
                        metrics[metric].append(value)

                # Média das métricas de todas as imagens
                for metric in base_metrics:

                    mean_value = np.mean(metrics[metric])

                    results[lp][metric].append(mean_value)

        # --------------------------------------------------
        # Exportação dos CSVs individuais por LP
        # --------------------------------------------------
        for lp in thr_list:

            df_lp = pd.DataFrame({
                "Length": Length_list
            })

            for metric in base_metrics:
                df_lp[metric] = results[lp][metric]

            lp_out_dir = os.path.join(
                length_out_dir,
                f"LP_{lp}"
            )
            os.makedirs(lp_out_dir, exist_ok=True)

            df_lp.to_csv(
                os.path.join(
                    lp_out_dir,
                    f"{band}_LP_{lp}_metrics.csv"
                ),
                index=False
            )

        # --------------------------------------------------
        # Exportação consolidada e gráficos
        # --------------------------------------------------
        for metric in base_metrics:

            df_all = pd.DataFrame({
                "Length": Length_list
            })

            for lp in thr_list:
                df_all[f"LP_{lp}"] = results[lp][metric]

            df_all.to_csv(
                os.path.join(
                    length_out_dir,
                    f"{band}_{metric}_LP_all.csv"
                ),
                index=False
            )

            plt.figure(figsize=(10, 6))

            for lp in thr_list:
                plt.plot(
                    Length_list,
                    results[lp][metric],
                    marker="o",
                    linewidth=2,
                    label=f"LP = {lp}%"
                )

            plt.xticks(ticks=range(0, len(Length_list) + 1, 50), rotation=45, ha="right")
            plt.xlabel("Length of structuring element")
            plt.ylabel(metric.replace("_", " ").upper())

            plt.title(
                f"{metric.replace('_', ' ').upper()} "
                f"vs Structuring Element Length"
            )

            plt.grid(True, alpha=0.3)
            plt.legend(
                title="Percentile threshold",
                ncol=2
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    length_out_dir,
                    f"{band}_{metric}_LP_all.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()
            
elif mm_operation == 'plot_graph':
    
    #abre o arquivo csv com a métrica a ser ilustrada
    
    
    #gera o gráfico
    df_all = pd.DataFrame({
        "Length": Length_list
    })

    for lp in thr_list:
        df_all[f"LP_{lp}"] = results[lp][metric]

    df_all.to_csv(
        os.path.join(
            length_out_dir,
            f"{band}_{metric}_LP_all.csv"
        ),
        index=False
    )

    plt.figure(figsize=(10, 6))

    for lp in thr_list:
        plt.plot(
            Length_list,
            results[lp][metric],
            marker="o",
            linewidth=2,
            label=f"LP = {lp}%"
        )

    plt.xticks(Length_list)
    plt.xlabel("Length of structuring element")
    plt.ylabel(metric.replace("_", " ").upper())

    plt.title(
        f"{metric.replace('_', ' ').upper()} "
        f"vs Structuring Element Length"
    )

    plt.grid(True, alpha=0.3)
    plt.legend(
        title="Percentile threshold",
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            length_out_dir,
            f"{band}_{metric}_LP_all.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    

# # #=================================================== old ===========================
#     # --------------------------------------------------
#    # Experimentos
#    # --------------------------------------------------
#    for band in band_list:
#        print(f"\nBanda: {band}")

#        # Armazena os resultados de todos os LPs
#        results = {
#            lp: {metric: [] for metric in all_metrics}
#            for lp in thr_list
#        }

#        for lp in thr_list:
#            print(f"\nLP: {lp}%")

#            for length in Length_list:
#                print(f"Comprimento B: {length}")

#                metrics = Length_Test(
#                    DS_DIR,
#                    band,
#                    mm_operation,
#                    B,
#                    length,
#                    skl_analysis=True,
#                    LP=lp
#                )

#                for metric in all_metrics:
#                    results[lp][metric].append(metrics[metric])

#            # --------------------------------------------------
#            # Salva um CSV individual para cada LP
#            # --------------------------------------------------
#            df_lp = pd.DataFrame({
#                "Length": Length_list
#            })

#            for metric in all_metrics:
#                df_lp[metric] = results[lp][metric]

#            lp_out_dir = os.path.join(
#                length_out_dir,
#                f"LP_{lp}"
#            )
#            os.makedirs(lp_out_dir, exist_ok=True)

#            df_lp.to_csv(
#                os.path.join(
#                    lp_out_dir,
#                    f"{band}_LP_{lp}_metrics.csv"
#                ),
#                index=False
#            )

#        # --------------------------------------------------
#        # Salva os resultados consolidados de todos os LPs
#        # --------------------------------------------------
#        for metric in all_metrics:

#            df_all = pd.DataFrame({
#                "Length": Length_list
#            })

#            for lp in thr_list:
#                df_all[f"LP_{lp}"] = results[lp][metric]

#            df_all.to_csv(
#                os.path.join(
#                    length_out_dir,
#                    f"{band}_{metric}_LP_all.csv"
#                ),
#                index=False
#            )

#            # --------------------------------------------------
#            # Gráfico com todos os LPs
#            # --------------------------------------------------
#            plt.figure(figsize=(10, 6))

#            for lp in thr_list:
#                plt.plot(
#                    Length_list,
#                    results[lp][metric],
#                    marker='o',
#                    linewidth=2,
#                    label=f'LP = {lp}%'
#                )

#            plt.xticks(Length_list[::10])
#            plt.xlabel("Length of structuring element")
#            plt.ylabel(metric.replace("_", " ").upper())

#            plt.title(
#                f"{metric.replace('_', ' ').upper()} "
#                f"vs Structuring Element Length"
#            )

#            plt.grid(True, alpha=0.3)
#            plt.legend(
#                title="Percentile threshold",
#                ncol=2
#            )

#            plt.tight_layout()

#            plt.savefig(
#                os.path.join(
#                    length_out_dir,
#                    f"{band}_{metric}_LP_all.png"
#                ),
#                dpi=300,
#                bbox_inches="tight"
#            )

#            plt.close()

