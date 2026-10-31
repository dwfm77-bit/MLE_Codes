#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:05:00 2025

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd
from skimage.morphology import skeletonize
import rasterio
from utils.mm_functions import (
     Morphological_Linear_Features_Enhance_S2_imgs_lite,
     Morphological_Linear_Features_Enhance_S2_imgs,
     plot_metrics_all_bands,
     Calc_Metrics,
     boundary_f1_score,
     gmm_filter_mask,
     metrics_dict_to_dataframe,
     Test_Length_process,
     summarize_length_results,
     ConditionalDilation,
     Test_SklThr_process, extract_metric,
     skeleton_analysis,
     linear_enhance
     )
from joblib import load
from itertools import combinations
import tifffile as tiff
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Setting the shape on a NumPy array has been deprecated.*",
    category=DeprecationWarning
)

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
band_list = ['ndwi']#['b2', 'b3', 'b4', 'b8', 'b11', 'b12', 'ndvi', 'ndbi', 'ndwi', 'ri']

dilation_list = ['b2','b3', 'b4', 'b11','b12','ndbi']
erosion_list = ['b8', 'ndvi', 'ndwi','ri']

gmm_mdl_path = "/home/usuario/Documentos/1.Doutorado/Testes/2. Testes_MLE/Versão_2/3. GMM_Model/GMM_Model/GMM_tests_Bands__b3_b8_b11"
#gmm_mdl_path = "/home/daniel/Documentos/1.Doutorado/Testes/2. Testes_MLE/Versão_2/3. GMM_Model/GMM_Model/GMM_tests_Bands__b3_b8_b11/"

gmm_bands = ['b3', 'b8', 'b11']

B = np.ones((3,3),np.int8)

Length_list = list(range(3, 513, 2))

thr_list = list(range(50, 100, 5))

linear_b_length = 101 # Valor final ajustado de B == 65

NS = [512,512]
mm_operation = ['erosion', 'dilation']

linear_enhancement = True # False/True
length_tests = False # False/True
Test_SklThr = False
cd_tests = False
nr_segment_tests = False

plot_metrics = False

if linear_enhancement:
    output_dir = output_dir / 'Linear_Enhancement_Tests'
elif length_tests:
    output_dir = output_dir / 'Length_Tests'
elif Test_SklThr:
    output_dir = output_dir / 'Skl_Threshold_Tests'
elif cd_tests:
    output_dir = output_dir / 'Conditional_Dilation_Tests'

#=========== CD Tests Parameters ===============
spatial_lim_grid = list(range(0, 11, 1))
# spectral_lim_grid = list(range(5, 100, 5))

# #Calcula as combinações dos limites espectrais
# spectral_lim_grid = list(combinations(spectral_lim_grid, 2))

#Combinacoes realizadas

# spectral_lim_grid = [(5, 10), (5, 15), (5, 20), (5, 25), (5, 30), (5, 35), (5, 40), (5, 45), (5, 50), (5, 55), (5, 60), (5, 65), (5, 70), (5, 75), (5, 80), (5, 85), (5, 90), (5, 95),
#                      (10, 15), (10, 20), (10, 25), (10, 30), (10, 35), (10, 40), (10, 45), (10, 50), (10, 55), (10, 60), (10, 65), (10, 70), (10, 75), (10, 80), (10, 85), (10, 90), (10, 95),
#                      (15, 20), (15, 25), (15, 30), (15, 35), (15, 40), (15, 45), (15, 50), (15, 55), (15, 60), (15, 65), (15, 70), (15, 75), (15, 80), (15, 85), (15, 90), (15, 95),
#                      (20, 25), (20, 30), (20, 35), (20, 40), (20, 45), (20, 50), (20, 55), (20, 60), (20, 65), (20, 70), (20, 75), (20, 80), (20, 85), (20, 90), (20, 95),
#                      (25, 30), (25, 35), (25, 40), (25, 45), (25, 50), (25, 55), (25, 60), (25, 65), (25, 70), (25, 75), (25, 80), (25, 85), (25, 90), (25, 95),
#                      (30, 35), (30, 40), (30, 45), (30, 50), (30, 55), (30, 60), (30, 65), (30, 70), (30, 75), (30, 80), (30, 85), (30, 90), (30, 95),
#                      (35, 40), (35, 45), (35, 50), (35, 55), (35, 60), (35, 65), (35, 70), (35, 75), (35, 80), (35, 85), (35, 90), (35, 95),
#                      (40, 45), (40, 50), (40, 55), (40, 60), (40, 65), (40, 70), (40, 75), (40, 80), (40, 85), (40, 90), (40, 95),
#                      (45, 50), (45, 55), (45, 60), (45, 65), (45, 70), (45, 75), (45, 80), (45, 85), (45, 90), (45, 95),
#                      (50, 55), (50, 60), (50, 65), (50, 70), (50, 75), (50, 80), (50, 85), (50, 90), (50, 95),
#                      (55, 60), (55, 65), (55, 70), (55, 75), (55, 80), (55, 85), (55, 90), (55, 95),
#                      (60, 65), (60, 70), (60, 75), (60, 80), (60, 85), (60, 90), (60, 95),
#                      (65, 70), (65, 75), (65, 80), (65, 85), (65, 90), (65, 95),
#                      (70, 75), (70, 80), (70, 85), (70, 90), (70, 95),
#                      (75, 80), (75, 85), (75, 90), (75, 95),
#                      (80, 85), (80, 90), (80, 95),
#                      (85, 90), (85, 95),
#                      (90, 95)]


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


if linear_enhancement:

    metrics_dilation = {}
    metrics_erosion  = {}
    
    Length_list = list(range(15, 19, 2))
    #Length_list = [35]
    
    #abre o modelo gmm para filtragem
    bundle = load(os.path.join(gmm_mdl_path,"gmm_model_best.joblib"))
    gmm_model = bundle["model"]
    
    
    print("Análise de realce linear\n")
        
    for b in band_list:
        print(f"Analisando banda {b}")

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
                cd_pctls = [5,50],
                sl = 2,
                save_results=True,
                show_imgs=False
            )


            metrics_dilation[b] = metrics
            
            # =============== Dilation Boxplots ==================
            
            metrics_list = ["iou", "f1", "precision", "recall", "bf_score"]
            
            plot_metrics_all_bands(metrics_dilation[b], metrics_list)
            
            
            extract_metric(metrics_dilation[b], "iou")
            extract_metric(metrics_dilation[b], "precision")
            extract_metric(metrics_dilation[b], "recall")
            extract_metric(metrics_dilation[b], "f1")
            extract_metric(metrics_dilation[b], "skl_iou")
            extract_metric(metrics_dilation[b], "skl_f1")
            extract_metric(metrics_dilation[b], "bf_score")
            extract_metric(metrics_dilation[b], "median_length_skl")
            
            df_dilation = metrics_dict_to_dataframe(metrics_dilation[b], "dilation")
            
            df_dilation.to_csv(
            output_dir / "metrics_linear_enhancement_dilation.csv",
            index=False
            )
            
            #Calcula as medianas do dataframe e gera um novo dataframe
            # Colunas cujas medianas serão calculadas
            metric_columns = [
                'iou', 'precision', 'recall', 'f1',
                'skl_iou', 'skl_precision', 'skl_recall',
                'skl_f1', 'bf_score'
            ]
            
            # Agrupar por length_B e calcular as medianas
            df_median = (
                df_dilation
                .groupby('length_B')[metric_columns]
                .median()
                .reset_index()
            )
                        
            #salva o dataframe em csv
            df_median.to_csv(
                output_dir / "median_metrics_linear_enhancement_dilation.csv",
                index=False
            )

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
                length_list = Length_list, #linear_b_length,
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
                cd_pctls = [5,50],
                sl = 2,
                save_results=True,
                show_imgs=False
            )

            metrics_erosion[b] = metrics
            
    
            # =============== Erosion Boxplots ==================
            
            metrics_list = ["iou", "f1", "precision", "recall", "bf_score"]
            
            plot_metrics_all_bands(metrics_erosion[b], metrics_list)

            extract_metric(metrics_erosion, "iou")
            extract_metric(metrics_erosion, "precision")
            extract_metric(metrics_erosion, "recall")
            extract_metric(metrics_erosion, "f1")
            extract_metric(metrics_erosion, "skl_iou")
            extract_metric(metrics_erosion, "skl_f1")
            extract_metric(metrics_erosion, "bf_score")
            extract_metric(metrics_erosion, "median_length_skl")
            
            df_erosion  = metrics_dict_to_dataframe(metrics_erosion, "erosion")
            
            #salva o dataframe em csv
            df_erosion.to_csv(
                output_dir / "metrics_linear_enhancement_erosion.csv",
                index=False
            )
            
            #Calcula as medianas do dataframe e gera um novo dataframe
            # Colunas cujas medianas serão calculadas
            metric_columns = [
                'iou', 'precision', 'recall', 'f1',
                'skl_iou', 'skl_precision', 'skl_recall',
                'skl_f1', 'bf_score'
            ]
            
            # Agrupar por length_B e calcular as medianas
            df_median = (
                df_erosion
                .groupby('length_B')[metric_columns]
                .median()
                .reset_index()
            )
                        
            #salva o dataframe em csv
            df_median.to_csv(
                output_dir / "median_metrics_linear_enhancement_erosion.csv",
                index=False
            )
#=======================================================================================

#Thr Tests
if Test_SklThr:
    thr_out_dir = output_dir
    os.makedirs(thr_out_dir, exist_ok=True)

    # Thr_Tests_Metrics_dilation = Test_SklThr_process(DS_DIR, thr_out_dir, linear_b_length, thr_list, dilation_list, 'dilation', B)

    # df_val_dil, df_len_dil = summarize_length_results(
    # Thr_Tests_Metrics_dilation,
    # Length_list,
    # dilation_list)
    
    Thr_Tests_Metrics_erosion = Test_SklThr_process(DS_DIR, thr_out_dir, linear_b_length, thr_list, erosion_list, 'erosion', B)
    
    df_val_ero, df_len_ero = summarize_length_results(
    Thr_Tests_Metrics_erosion,
    Length_list,
    erosion_list)
    
    
    # out_csv_val = os.path.join(
    #     thr_out_dir, "thr_tests_best_metric_value_dilation.csv"
    # )
    # out_csv_len = os.path.join(
    #     thr_out_dir, "thr_tests_best_length_dilation.csv"
    #)

    # df_val_dil.to_csv(out_csv_val, sep=";", decimal=",", encoding="utf-8-sig")
    # df_len_dil.to_csv(out_csv_len, sep=";", decimal=",", encoding="utf-8-sig")
    
    
    out_csv_val = thr_out_dir / "thr_tests_best_metric_value_erosion.csv"
    out_csv_len = thr_out_dir / "thr_tests_best_length_erosion.csv"
    

    df_val_ero.to_csv(out_csv_val, sep=";", decimal=",", encoding="utf-8-sig")
    df_len_ero.to_csv(out_csv_len, sep=";", decimal=",", encoding="utf-8-sig")
    
    # #Salva os resultados finais do teste de dilatação em xlsx
    # history_dilation = os.path.join(thr_out_dir, "thr_tests_dilation.xlsx")
    # with pd.ExcelWriter(history_dilation, engine="openpyxl") as writer:

    #    for metric in Thr_Tests_Metrics_dilation.keys():

    #        df = pd.DataFrame(
    #            {"Threshold": thr_list}
    #        )

    #        for band in Thr_Tests_Metrics_dilation[metric]:
    #            df[str(band)] = Thr_Tests_Metrics_dilation[metric][band]

    #        df.to_excel(
    #            writer,
    #            sheet_name=metric[:31],  # limite do Excel
    #            index=False
    #        )
    
    #Salva os resultados finais do teste de erosão em xlsx
    history_erosion = thr_out_dir / "thr_tests_erosion.xlsx"
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

#Length Tests
if length_tests:
    output_dir = '/home/usuario/Documentos/1.Doutorado/Datasets/temp'
    length_out_dir = output_dir / 'Length_Tests'
    os.makedirs(length_out_dir, exist_ok=True)


    Length_Tests_Metrics_dilation = Test_Length_process(DS_DIR, length_out_dir, Length_list, dilation_list, 'dilation', B, skl_a = False)
    
    df_val_dil, df_len_dil = summarize_length_results(
    Length_Tests_Metrics_dilation,
    Length_list,
    dilation_list)
    
    Length_Tests_Metrics_erosion = Test_Length_process(DS_DIR, length_out_dir, Length_list, erosion_list, 'erosion', B, skl_a = False)
    
    df_val_ero, df_len_ero = summarize_length_results(
    Length_Tests_Metrics_erosion,
    Length_list,
    erosion_list)
    
    
    out_csv_val = length_out_dir / "length_tests_best_metric_value_dilation.csv"
    out_csv_len = length_out_dir / "length_tests_best_length_dilation.csv"


    df_val_dil.to_csv(out_csv_val, sep=";", decimal=",", encoding="utf-8-sig")
    df_len_dil.to_csv(out_csv_len, sep=";", decimal=",", encoding="utf-8-sig")
    
    
    out_csv_val = length_out_dir / "length_tests_best_metric_value_erosion.csv"
    out_csv_len = length_out_dir / "length_tests_best_length_erosion.csv"

    df_val_ero.to_csv(out_csv_val, sep=";", decimal=",", encoding="utf-8-sig")
    df_len_ero.to_csv(out_csv_len, sep=";", decimal=",", encoding="utf-8-sig")
    
    #Salva os resultados finais do teste de dilatação em xlsx
    history_dilation = length_out_dir / "length_tests_dilation.xlsx"
    with pd.ExcelWriter(history_dilation, engine="openpyxl") as writer:

       for metric in Length_Tests_Metrics_dilation.keys():

           df = pd.DataFrame(
               {"Length": Length_list}
           )

           for band in Length_Tests_Metrics_dilation[metric]:
               df[str(band)] = Length_Tests_Metrics_dilation[metric][band]

           df.to_excel(
               writer,
               sheet_name=metric[:31],  # limite do Excel
               index=False
           )
    
    # Salva os resultados finais dos testes de dilatação em CSV
    for metric, bands in Length_Tests_Metrics_dilation.items():
    
        df = pd.DataFrame({"Length": Length_list})
    
        for band, values in bands.items():
            df[str(band)] = values
    
        out_csv = length_out_dir/ f"length_tests_dilation_{metric}.csv"
    
        df.to_csv(
            out_csv,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
            index=False
        )
    
    #Salva os resultados finais do teste de erosão em xlsx
    history_erosion = length_out_dir / "length_tests_erosion.xlsx"
    with pd.ExcelWriter(history_erosion, engine="openpyxl") as writer:

       for metric in Length_Tests_Metrics_erosion.keys():

           df = pd.DataFrame(
               {"Length": Length_list}
           )

           for band in Length_Tests_Metrics_erosion[metric]:
               df[str(band)] = Length_Tests_Metrics_erosion[metric][band]

           df.to_excel(
               writer,
               sheet_name=metric[:31],  # limite do Excel
               index=False
           )
    
    
    # Salva os resultados finais dos testes de erosão em CSV
    for metric, bands in Length_Tests_Metrics_erosion.items():
    
        df = pd.DataFrame({"Length": Length_list})
    
        for band, values in bands.items():
            df[str(band)] = values
    
        out_csv = length_out_dir / f"length_tests_erosion_{metric}.csv"
    
        df.to_csv(
            out_csv,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
            index=False
        )



if cd_tests:

    metrics_dilation = {}
    metrics_erosion  = {}

    length_b = 101
    
    band_cd = 'ndwi'#['ri']
    
    #abre o modelo gmm para filtragem
    bundle = load(os.path.join(gmm_mdl_path,"gmm_model_best.joblib"))
    gmm_model = bundle["model"]
    
    # gmm_model = load(os.path.join(gmm_mdl_path,"gmm_model_best.joblib"))
    # scaler_ds = load(os.path.join(gmm_mdl_path,"gmm_scaler_best.joblib"))
    
    temp_imgs_dir = output_dir / 'Temp_Imgs'
    os.makedirs(temp_imgs_dir, exist_ok=True),
    
    print("Salva as imagens realçadas\n")
    # for band_cd in band_list:
    #     print(f"Analisando banda {band_cd}")
        
    #     out_dir = temp_imgs_dir / 'band_cd'
    #     os.makedirs(temp_imgs_dir, exist_ok=True),

    #     if band_cd in dilation_list:

    #         # -------------------------
    #         # Dilatação
    #         # -------------------------
    #         mm_operator = mm_operation[1]
    
    #         metrics = Morphological_Linear_Features_Enhance_S2_imgs_lite(
    #             img_dir=DS_DIR,
    #             output_dir= out_dir,
    #             band=band_cd,
    #             operacao=mm_operator,
    #             length_b = length_b,
    #             NS=NS,
    #             ste=B,
    #             thickness=1,
    #             p_processing=True,
    #             channel_gmm=gmm_bands,
    #             skl_analysis=True,
    #             gmm_analysis=True,
    #             gmm_mdl=gmm_model,
    #             conditional_dil=False,
    #             save_results=True,
    #             show_imgs=False
    #         )
        
    #     else:
    #         # -------------------------
    #         # Erosão
    #         # -------------------------
    #         mm_operator = mm_operation[0]
    
    #         metrics = Morphological_Linear_Features_Enhance_S2_imgs_lite(
    #             img_dir=DS_DIR,
    #             output_dir= out_dir,
    #             band=band_cd,
    #             operacao=mm_operator,
    #             length_b = length_b, #linear_b_length,
    #             NS=NS,
    #             ste=B,
    #             thickness=1,
    #             p_processing=False,
    #             lp=95,
    #             channel_gmm=gmm_bands,
    #             skl_analysis=True,
    #             gmm_analysis=True,
    #             gmm_mdl=gmm_model,
    #             conditional_dil=False,
    #             save_results=True,
    #             show_imgs=False
    #         )

    
    
    print("Análise dos limiares percentil e espacial\n")
    
    # Acumuladores globais de métricas
    metrics_dilation_rows = []
    metrics_erosion_rows = []
    
    
    # Acumuladores das medianas por combinação LP/SL
    median_dilation_rows = []
    median_erosion_rows = []
    
    best_metrics_ero = {
        "LP":0,
        "SL":0,
        "iou":0,
        "precision":0,
        "recall": 0,
        "f1": 0,
        "skl_iou": 0,
        "bf_score": 0
        }
    
    best_metrics_dil = {
        "LP":0,
        "SL":0,
        "iou":0,
        "precision":0,
        "recall": 0,
        "f1": 0,
        "skl_iou": 0,
        "bf_score": 0
        }

    for LP in spectral_lim_grid:
        #print(f"Limiar Percentil: {LP}")
        
        temp_path = output_dir / f"LP: {LP}"
        
        if not os.path.exists(temp_path):
            os.makedirs(temp_path, exist_ok=True)
        
            for SL in spatial_lim_grid:
            
                #print(f"Limiar Espacial: {SL}")    
    
                if band_cd in dilation_list:
                    # -------------------------
                    # Dilatação
                    # -------------------------
                    mm_operator = mm_operation[1]
            
                    band_dir = img_dir / band_cd
                    image_files = sorted([
                        f for f in os.listdir(band_dir)
                        if f.lower().endswith((".tif", ".tiff"))
                    ])
                    
                    metrics_hist = {
                        "LP":[],
                        "img_nm":[],
                        "iou": [],
                        "precision": [],
                        "recall": [],
                        "f1": [],
                        "skl_iou": [],
                        "bf_score": []
                        }
                    
                    cont = 1
                    
                    #for img in image_files:
                    for img in tqdm(image_files, desc=f"LP: {LP} – SL: {SL}"):
                        
                        #abre a imagem a ser realçada            
                        # img_path = os.path.join(band_dir, img)
                        # with rasterio.open(img_path) as src:
                        #     img_b = src.read(1)
                        #     meta = src.meta.copy()
                            
                        img_b = np.array(Image.open(band_dir / img).convert("L")) > 0
                            
                        #aplica a filtragem por GMM
                        if img_b.max() > 1:
                            img_s = (img_b - img_b.min()) / (img_b.max() - img_b.min() + 1e-8)
                            
                        #Abre as bandas compostas para filtragem por GMM
                        img_s = None
                        meta = None
                        
                        for i, ch in enumerate(gmm_bands):
                            img_path = img_dir / "Imgs" / ch / img
                        
                            with rasterio.open(img_path) as src:
                                band_img = src.read(1).astype(np.float32)
                        
                                # Inicialização do array empilhado
                                if img_s is None:
                                    H, W = band_img.shape
                                    img_s = np.zeros((len(gmm_bands), H, W), dtype=np.float32)
                        
                                    # Metadados de referência (usar sempre a primeira banda)
                                    meta = src.meta.copy()
                                else:
                                    # Verificação de consistência espacial
                                    assert band_img.shape == (H, W), f"Incompatibilidade espacial em {ch}/{img}"
                                    assert src.transform == meta["transform"], f"Transform diferente em {ch}/{img}"
                                    assert src.crs == meta["crs"], f"CRS diferente em {ch}/{img}"
                        
                                img_s[i] = band_img  
                        
                        #filtragem por GMM
                        img_gmm = gmm_filter_mask(img_s, img_b, gmm_model)
                            
                        #aplica o operador de dilatacao condicional
                        rec_img = ConditionalDilation(img_gmm, img_b, pctls=LP, m_dist=SL)
                        
                        #calcula as metricas
                        img_bin = (rec_img > 0).astype(np.uint8)
            
                        # imagem de saída (uint8) APENAS para salvamento
                        img_out = (img_bin * 255)
                        
                        #abre a mascara de referencia
                        mask_ref = np.array(Image.open(DS_DIR / "Mask" / img)).convert("L")) > 0
                        
                        # --------------------------------------------------
                        # Métricas
                        # --------------------------------------------------
                        (iou, precision, recall, f1, _, _, _, _) = Calc_Metrics(img_bin, mask_ref)
            
                        #bf-score
                        bf_score = boundary_f1_score(mask_ref, img_bin, tolerance=2)
            
                        # --------------------------------------------------
                        # Métricas sobre esqueletos
                        # --------------------------------------------------
            
                        skl_mask_ref = skeletonize(mask_ref, method="zhang")
                        #skl_img_bin = skeletonize(img_bin, method="zhang")
            
                        # if img_bin.sum() == 0:
                        #     print(f"[WARN] Esqueleto previsto vazio: {f}")
                        
                        # if skl_mask_ref.sum() == 0:
                        #     print(f"[WARN] Esqueleto GT vazio: {f}")
                        
                        (skl_iou, _, _, _, _, _, _, _) = Calc_Metrics(img_bin, skl_mask_ref)
                        
                        metrics_hist["LP"].append(LP)
                        metrics_hist["img_nm"].append(img)
                        metrics_hist["iou"].append(iou)
                        metrics_hist["f1"].append(f1)
                        metrics_hist["precision"].append(precision)
                        metrics_hist["recall"].append(recall)
                        metrics_hist["skl_iou"].append(skl_iou)
                        metrics_hist["bf_score"].append(bf_score)
                        
                    # --------------------------------------------------
                    # Salva as métricas individuais das imagens
                    # para a combinação atual de LP e SL
                    # --------------------------------------------------
                    
                    metrics_hist["SL"] = [SL] * len(metrics_hist["LP"])
                    
                    df_metrics = pd.DataFrame(metrics_hist)
                    
                    csv_name = f"metrics_LP_{LP}_SL_{SL}.csv"
                    
                    csv_path = temp_path / csv_name
                    
                    df_metrics.to_csv(
                        csv_path,
                        index=False,
                        encoding="utf-8-sig",
                        sep=";"
                    )
                    
                    #Calcula das medianas das metricas
                    iou_median = np.median(metrics_hist['iou'])
                    f1_median = np.median(metrics_hist['f1'])
                    prec_median = np.median(metrics_hist['precision'])
                    rec_median = np.median(metrics_hist['recall'])
                    skl_iou_median = np.median(metrics_hist['skl_iou'])
                    bf_median = np.median(metrics_hist['bf_score'])
                    
                    
                    # Registra as medianas da combinação atual de LP e SL
                    median_dilation_rows.append({
                        "LP": LP,
                        "SL": SL,
                        "iou_median": iou_median,
                        "precision_median": prec_median,
                        "recall_median": rec_median,
                        "f1_median": f1_median,
                        "skl_iou_median": skl_iou_median,
                        "bf_score_median": bf_median
                    })
                    
                    # --------------------------------------------------
                    # Identificação do melhor resultado por IoU
                    # --------------------------------------------------                
                    if best_metrics_dil["iou"] == 0:
                        
                        best_metrics_dil["LP"] = LP
                        best_metrics_dil["SL"] = SL
                        best_metrics_dil["iou"] = iou_median
                        best_metrics_dil["precision"] = prec_median
                        best_metrics_dil["recall"] = rec_median
                        best_metrics_dil["f1"] = f1_median
                        best_metrics_dil["skl_iou"] = skl_iou_median
                        best_metrics_dil["bf_score"] = bf_median
                        
                    elif best_metrics_dil["iou"] > iou_median:
                    
                        best_metrics_dil["LP"] = LP
                        best_metrics_dil["SL"] = SL
                        best_metrics_dil["iou"] = iou_median
                        best_metrics_dil["precision"] = prec_median
                        best_metrics_dil["recall"] = rec_median
                        best_metrics_dil["f1"] = f1_median
                        best_metrics_dil["skl_iou"] = skl_iou_median
                        best_metrics_dil["bf_score"] = bf_median
    
                    # Registra o limiar espacial de cada resultado
                    metrics_hist["SL"] = [SL] * len(metrics_hist["LP"])
                    
                    # Acumula os resultados da combinação LP/SL
                    metrics_dilation_rows.extend(
                        pd.DataFrame(metrics_hist).to_dict("records")
                    )
            
                else:
                    # -------------------------
                    # Erosão
                    # -------------------------
                    mm_operator = mm_operation[0]
            
                    band_dir = img_dir / band_cd
                    image_files = sorted([
                        f for f in os.listdir(band_dir)
                        if f.lower().endswith((".tif", ".tiff"))
                    ])
                    
                    metrics_hist = {
                        "LP":[],
                        "img_nm":[],
                        "iou": [],
                        "precision": [],
                        "recall": [],
                        "f1": [],
                        "skl_iou": [],
                        "bf_score": []
                        }
                    
                    for img in tqdm(image_files, desc=f"LP: {LP} – SL: {SL}"):
                        
                        #abre a imagem a ser realçada            
                        # img_path = os.path.join(band_dir, img)
                        # with rasterio.open(img_path) as src:
                        #     img_b = src.read(1)
                        #     meta = src.meta.copy()
                            
                        img_b = np.array(Image.open(band_dir/ img))
                            
                        #aplica a filtragem por GMM
                        # if img_b.max() > 1:
                        #     img_b = (img_b - img_b.min()) / (img_b.max() - img_b.min() + 1e-8)
                            
                        #Abre as bandas compostas para filtragem por GMM
                        img_s = None
                        meta = None
                        
                        for i, ch in enumerate(gmm_bands):
                            img_path = img_dir / ch / img
                        
                            with rasterio.open(img_path) as src:
                                band_img = src.read(1).astype(np.float32)
                        
                                # Inicialização do array empilhado
                                if img_s is None:
                                    H, W = band_img.shape
                                    img_s = np.zeros((len(gmm_bands), H, W), dtype=np.float32)
                        
                                    # Metadados de referência (usar sempre a primeira banda)
                                    meta = src.meta.copy()
                                else:
                                    # Verificação de consistência espacial
                                    assert band_img.shape == (H, W), f"Incompatibilidade espacial em {ch}/{img}"
                                    assert src.transform == meta["transform"], f"Transform diferente em {ch}/{img}"
                                    assert src.crs == meta["crs"], f"CRS diferente em {ch}/{img}"
                        
                                img_s[i] = band_img
                                
                        if img_s.max() > 1:
                            img_s = (img_s - img_s.min()) / (img_s.max() - img_s.min() + 1e-8)
                        
                        #filtragem por GMM
                        img_gmm = gmm_filter_mask(img_s, img_b, gmm_model)
                        
                        #aplica o operador de dilatacao condicional
                        rec_img = ConditionalDilation(img_gmm, img_b, pctls=LP, m_dist=SL)
                        
                        #calcula as metricas
                        img_bin = (rec_img > 0).astype(np.uint8)
            
                        # imagem de saída (uint8) APENAS para salvamento
                        img_out = (img_bin * 255)
                        
                        #abre a mascara de referencia
                        mask_ref = np.array(Image.open(DS_DIR / "Mask", img).convert("L")) > 0
                        
                        # --------------------------------------------------
                        # Métricas
                        # --------------------------------------------------
                        
                        (iou, precision, recall, f1, _, _, _, _) = Calc_Metrics(img_bin, mask_ref)
            
                        #bf-score
                        bf_score = boundary_f1_score(mask_ref, img_bin, tolerance=2)
            
                        # --------------------------------------------------
                        # Métricas sobre esqueletos
                        # --------------------------------------------------
            
                        skl_mask_ref = skeletonize(mask_ref, method="zhang")
                        #skl_img_bin = skeletonize(img_bin, method="zhang")
            
                        # if img_bin.sum() == 0:
                        #     print(f"[WARN] Esqueleto previsto vazio: {f}")
                        
                        # if skl_mask_ref.sum() == 0:
                        #     print(f"[WARN] Esqueleto GT vazio: {f}")
                        
                        (skl_iou, _, _, _, _, _, _, _) = Calc_Metrics(img_bin, skl_mask_ref)
                        
                        metrics_hist["LP"].append(LP)
                        metrics_hist["img_nm"].append(img)
                        metrics_hist["iou"].append(iou)
                        metrics_hist["f1"].append(f1)
                        metrics_hist["precision"].append(precision)
                        metrics_hist["recall"].append(recall)
                        metrics_hist["skl_iou"].append(skl_iou)
                        metrics_hist["bf_score"].append(bf_score)
                        
                    # --------------------------------------------------
                    # Salva as métricas individuais das imagens
                    # para a combinação atual de LP e SL
                    # --------------------------------------------------
                    
                    metrics_hist["SL"] = [SL] * len(metrics_hist["LP"])
                    
                    df_metrics = pd.DataFrame(metrics_hist)
                    
                    csv_name = f"metrics_LP_{LP}_SL_{SL}.csv"
                    
                    csv_path = temp_path / csv_name
                    
                    df_metrics.to_csv(
                        csv_path,
                        index=False,
                        encoding="utf-8-sig",
                        sep=";"
                    )
                    
                        
                    #Calcula das medianas das metricas
                    iou_median = np.median(metrics_hist['iou'])
                    f1_median = np.median(metrics_hist['f1'])
                    prec_median = np.median(metrics_hist['precision'])
                    rec_median = np.median(metrics_hist['recall'])
                    skl_iou_median = np.median(metrics_hist['skl_iou'])
                    bf_median = np.median(metrics_hist['bf_score'])
                    
                    # Registra as medianas da combinação atual de LP e SL
                    median_erosion_rows.append({
                        "LP": LP,
                        "SL": SL,
                        "iou_median": iou_median,
                        "precision_median": prec_median,
                        "recall_median": rec_median,
                        "f1_median": f1_median,
                        "skl_iou_median": skl_iou_median,
                        "bf_score_median": bf_median
                    })
                                    
                    # --------------------------------------------------
                    # Identificação do melhor resultado por IoU
                    # --------------------------------------------------                
                    if best_metrics_ero["iou"] == 0:
                        
                        best_metrics_ero["LP"] = LP
                        best_metrics_ero["SL"] = SL
                        best_metrics_ero["iou"] = iou_median
                        best_metrics_ero["precision"] = prec_median
                        best_metrics_ero["recall"] = rec_median
                        best_metrics_ero["f1"] = f1_median
                        best_metrics_ero["skl_iou"] = skl_iou_median
                        best_metrics_ero["bf_score"] = bf_median
                        
                    elif best_metrics_ero["iou"] > iou_median:
                    
                        best_metrics_ero["LP"] = LP
                        best_metrics_ero["SL"] = SL
                        best_metrics_ero["iou"] = iou_median
                        best_metrics_ero["precision"] = prec_median
                        best_metrics_ero["recall"] = rec_median
                        best_metrics_ero["f1"] = f1_median
                        best_metrics_ero["skl_iou"] = skl_iou_median
                        best_metrics_ero["bf_score"] = bf_median
            
                    # Registra o limiar espacial de cada resultado
                    metrics_hist["SL"] = [SL] * len(metrics_hist["LP"])
                    
                    # Acumula os resultados da combinação LP/SL
                    metrics_erosion_rows.extend(
                        pd.DataFrame(metrics_hist).to_dict("records")
                    )
                
    
                # --------------------------------------------------
                # Exibição e salvamento das melhores métricas
                # --------------------------------------------------
                
                # Organiza os resultados
                df_best_dil = pd.DataFrame([best_metrics_dil])
                df_best_ero = pd.DataFrame([best_metrics_ero])
                
                # # Exibe os resultados no console
                # print("\nMelhores métricas - Dilatação:")
                # print(df_best_dil.to_string(index=False))
                
                # print("\nMelhores métricas - Erosão:")
                # print(df_best_ero.to_string(index=False))
                
                
                # Salva os arquivos CSV
                os.makedirs(output_dir, exist_ok=True)
                
                df_best_dil.to_csv(output_dir / "best_metrics_dilation.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";"
                )
                
                df_best_ero.to_csv(output_dir / "best_metrics_erosion.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";"
                )
                
                # print("\nArquivos CSV salvos com sucesso:")
                # print(os.path.join(output_dir, "best_metrics_dilation.csv"))
                # print(os.path.join(output_dir, "best_metrics_erosion.csv"))
                
                
                if plot_metrics:
                    #plota as métricas
                    if band_cd in dilation_list:
                        
                        # =============== Dilation Boxplots ==================
                        
                        plot_metrics_all_bands(
                            metrics_dict={b: metrics_dilation[b]["iou"] for b in metrics_dilation},
                            metric_name="IoU",
                            out_dir = temp_path,
                            title=f"IoU – MLED - CD - LP: {LP} - SL: {SL}",
                            ylabel="IoU",
                            show_stats=False
                        )
                        
                        plot_metrics_all_bands(
                            metrics_dict={b: metrics_dilation[b]["f1"] for b in metrics_dilation},
                            metric_name="F1-Score",
                            out_dir = temp_path,
                            title=f"F1-Score – MLED - CD - LP: {LP} - SL: {SL}",
                            ylabel="F1-Score",
                            show_stats=False
                        )
                        
                        plot_metrics_all_bands(
                            metrics_dict={b: metrics_dilation[b]["skl_iou"] for b in metrics_dilation},
                            metric_name="SKL-IoU",
                            out_dir = temp_path,
                            title= f"Skl IoU – MLED - CD - LP: {LP} - SL: {SL}",
                            ylabel="Skl IoU",
                            show_stats=False
                        )
                            
                        extract_metric(metrics_dilation, "iou")
                        extract_metric(metrics_dilation, "precision")
                        extract_metric(metrics_dilation, "recall")
                        extract_metric(metrics_dilation, "f1")
                        extract_metric(metrics_dilation, "skl_iou")
                        extract_metric(metrics_dilation, "bf_score")
                        
                        df_dilation = metrics_dict_to_dataframe(metrics_dilation, "dilation")
                        
                        df_dilation.to_csv(
                        temp_path / f"metrics_MLED_LP: {LP} - SL: {SL}.csv",
                        index=False
                        )
        
                        
                    else:
                    
                    
                        # =============== Erosion Boxplots ==================
                        
                        plot_metrics_all_bands(
                            metrics_dict={b: metrics_erosion[b]["iou"] for b in metrics_erosion},
                            metric_name="IoU",
                            out_dir = temp_path,
                            title=f"IoU – MLEE - CD - LP: {LP} - SL: {SL}",
                            ylabel="IoU",
                            show_stats=False
                        )
                        
                        plot_metrics_all_bands(
                            metrics_dict={b: metrics_erosion[b]["f1"] for b in metrics_erosion},
                            metric_name="F1-Score",
                            out_dir = temp_path,
                            title=f"F1-Score – MLEE - CD - LP: {LP} - SL: {SL}",
                            ylabel="F1-Score",
                            show_stats=False
                        )
                        
                        plot_metrics_all_bands(
                            metrics_dict={b: metrics_erosion[b]["skl_iou"] for b in metrics_erosion},
                            metric_name="SKL-IoU",
                            out_dir = temp_path,
                            title=f"Skl IoU – MLEE - CD - LP: {LP} - SL: {SL}",
                            ylabel="Skl IoU",
                            show_stats=False
                        )
                    
                        extract_metric(metrics_erosion, "iou")
                        extract_metric(metrics_erosion, "precision")
                        extract_metric(metrics_erosion, "recall")
                        extract_metric(metrics_erosion, "f1")
                        extract_metric(metrics_erosion, "skl_iou")
                        extract_metric(metrics_erosion, "bf_score")
                        
                        df_erosion  = metrics_dict_to_dataframe(metrics_erosion, "erosion")
                        
                        df_erosion.to_csv(temp_path / f"metrics_MLEE_LP: {LP} - SL: {SL}.csv",
                            index=False
                        )
    


    # --------------------------------------------------
    # Salva as medianas de todas as combinações LP/SL
    # --------------------------------------------------
    
    os.makedirs(output_dir, exist_ok=True)
    
   # Medianas da dilatação
    df_median_dilation = pd.DataFrame(median_dilation_rows)
    
    file_dilation = output_dir / "median_metrics_dilation.csv"
    
    if os.path.exists(file_dilation) and os.path.getsize(file_dilation) > 0:
        try:
            df_previous = pd.read_csv(
                file_dilation,
                sep=";",
                encoding="utf-8-sig"
            )
    
            df_median_dilation = pd.concat(
                [df_previous, df_median_dilation],
                ignore_index=True
            )
    
        except pd.errors.EmptyDataError:
            print(
                f"[AVISO] O arquivo {file_dilation} está vazio. "
                "Será preenchido com os novos dados."
            )
    
    df_median_dilation.to_csv(
        file_dilation,
        index=False,
        encoding="utf-8-sig",
        sep=";"
    )
    
    df_median_dilation.to_csv(
        file_dilation,
        index=False,
        encoding="utf-8-sig",
        sep=";"
    )
    
    
    # Medianas da erosão
    df_median_erosion = pd.DataFrame(median_erosion_rows)
    
    file_erosion = output_dir / "median_metrics_erosion.csv"
    
    if os.path.exists(file_erosion) and os.path.getsize(file_erosion) > 0:
        try:
            df_previous = pd.read_csv(
                file_erosion,
                sep=";",
                encoding="utf-8-sig"
            )
    
            df_median_erosion = pd.concat(
                [df_previous, df_median_erosion],
                ignore_index=True
            )
    
        except pd.errors.EmptyDataError:
            print(
                f"[AVISO] O arquivo {file_erosion} está vazio. "
                "Será preenchido com os novos dados."
            )
    
    df_median_erosion.to_csv(
        file_erosion,
        index=False,
        encoding="utf-8-sig",
        sep=";"
    )

    
    print("Arquivos de medianas salvos:")
    print(output_dir / "median_metrics_dilation.csv")
    print(output_dir / "median_metrics_erosion.csv")
    
if nr_segment_tests:

    output_dir = '/home/usuario/Documentos/1.Doutorado/Testes/Temp/Segments_Analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    Length_list = list(range(3, 513, 2))
    m_operator = 'erosion'
    band_list = ['ri']#['b8', 'ndvi', 'ndwi', 'ri']
    # dilation_list = ['b2','b3', 'b4', 'b11','b12','ndbi']
    # erosion_list = ['b8', 'ndvi', 'ndwi','ri']
    
    metrics_per_band = {
        ch: {
            'lengths': [],
            'mean_length': [],
            'median_length': [],
            'median_n_segments': [],
            'n_selected': []
        }
        for ch in band_list
    }
    
    
    imgs_files = os.listdir(mask_dir) # substituir pelo folder das predições
    imgs_files = [f for f in imgs_files if f.lower().endswith(('.tif', '.tiff'))]
    
    for length in Length_list:
        print(f"- Comprimento B: {length}")

        for ch in band_list:
            
            mean_lengths_img   = []
            median_lengths_img = []
            n_segments_img     = []
            n_selectec_segments = []
                        
            for img_name in tqdm(imgs_files, desc=f"{ch} | B={length}"):
            
                # --- Máscara GT ---
                mask_path = mask_dir / img_name
                mask_gt = tiff.imread(mask_path).astype(np.uint8)
                    
                img_path = img_dir / ch / img_name
                img_1ch = tiff.imread(img_path).astype(np.float32)
                    
                # # Pré-processamento opcional
                # img_b = cv2.GaussianBlur(img_1ch, (5,5), 0)
                img_b = img_1ch
                    
                # #plt.title(f"Img_Pre-processing {f}")
                # plt.imshow(img_b, cmap="gray")
                # plt.axis("off")
                # plt.show()
        
                # --------------------------------------------------
                # Realce linear
                # --------------------------------------------------
                img_enhanced = linear_enhance(img_b, length, m_operator, thickness=1)
                
                # --------------------------------------------------
                # Pós-processamento estrutural
                # --------------------------------------------------
                img_skl, img_skl_stats = skeleton_analysis(img_enhanced, ste=B, seg_filter = True, length_percentile=85)

                mean_lengths_img.append(img_skl_stats['mean_length'])
                median_lengths_img.append(img_skl_stats['median_length'])
                n_segments_img.append(img_skl_stats['n_skeletons'])
                n_selectec_segments.append(img_skl_stats['n_selected'])
                
            # ---- agregação por comprimento B ----
            metrics_per_band[ch]['lengths'].append(length)
            metrics_per_band[ch]['mean_length'].append(np.mean(mean_lengths_img))
            metrics_per_band[ch]['median_length'].append(np.median(median_lengths_img))
            metrics_per_band[ch]['median_n_segments'].append(np.median(n_segments_img))
            metrics_per_band[ch]['n_selected'].append(np.average(n_selectec_segments))
            
    #Gráfico das médias de segmentos por banda - Inglês
    plt.figure()
    for ch in band_list:
        plt.plot(
            metrics_per_band[ch]['lengths'],
            metrics_per_band[ch]['median_n_segments'],
            label=ch
        )
    
    plt.xlabel("Length of the structuring element |B|")
    plt.ylabel("Average number of segments")
    plt.title("Number of segments vs. |B|")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "Median_Number_of_Segments.png", dpi = 200)
    plt.show()
    
    #Gráfico da média de comprimento dos segmentos por banda
    plt.figure()
    for ch in band_list:
        plt.plot(
            metrics_per_band[ch]['lengths'],
            metrics_per_band[ch]['mean_length'],
            label=ch
        )

    plt.xlabel("Length of the structuring element |B|")
    plt.ylabel("Average segment length")
    plt.title("Average segment length vs. |B|")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "Average_Segment_Length.png", dpi = 200)
    plt.show()
    
    
    #Gráfico da média de nr de segmentos selecionados por banda
    plt.figure()
    for ch in band_list:
        plt.plot(
            metrics_per_band[ch]['lengths'],
            metrics_per_band[ch]['n_selected'],
            label=ch
        )

    plt.xlabel("Length of the structuring element |B|")
    plt.ylabel("Average number of selected segments")
    plt.title("Average number of selected elements vs.|B|")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "Average_Selected_Segments.png", dpi = 200)
    plt.show()
    
    
    
    #Gráfico das médias de segmentos por banda - Português
    plt.figure()
    for ch in band_list:
       plt.plot(
           metrics_per_band[ch]['lengths'],
           metrics_per_band[ch]['median_n_segments'],
           label=ch
       )
   
    plt.xlabel("Comprimento do elemento estruturante |B|")
    plt.ylabel("Número médio de segmentos")
    plt.title("Número de segmentos vs |B|")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "Mediana_Nr_Segmentos.png", dpi = 200)
    plt.show()
   
    #Gráfico da média de comprimento dos segmentos por banda
    plt.figure()
    for ch in band_list:
        plt.plot(
            metrics_per_band[ch]['lengths'],
            metrics_per_band[ch]['mean_length'],
            label=ch
        )

    plt.xlabel("Comprimento do elemento estruturante |B|")
    plt.ylabel("Comprimento médio dos segmentos")
    plt.title("Média do comprimento dos segmentos vs |B|")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "Media_Comprimento_Segmentos.png", dpi = 200)
    plt.show()
    
    
    
    #Gráfico da média de nr de segmentos selecionados por banda
    plt.figure()
    for ch in band_list:
        plt.plot(
            metrics_per_band[ch]['lengths'],
            metrics_per_band[ch]['n_selected'],
            label=ch
        )

    plt.xlabel("Comprimento do elemento estruturante |B|")
    plt.ylabel("Nr médio de segmentos")
    plt.title("Nr médio de elementos selecionados vs. |B|")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "Media_segmentos_selecionados.png", dpi = 200)
    plt.show()
        
    
    #salva os dados em um .csv    
    rows = []

    for ch in band_list:
        for i, B in enumerate(metrics_per_band[ch]['lengths']):
            rows.append({
                "band": ch,
                "B_length": B,
                "mean_segment_length": metrics_per_band[ch]['mean_length'][i],
                "median_segment_length": metrics_per_band[ch]['median_length'][i],
                "median_number_of_segments": metrics_per_band[ch]['median_n_segments'][i],
                "average_selected_segments": metrics_per_band[ch]['n_selected'][i]
            })
    
    df_metrics = pd.DataFrame(rows)
    
    csv_path = output_dir / "segment_metrics_by_band.csv")
    df_metrics.to_csv(csv_path, index=False)
    
    
    print('Processamento de análise de comprimentos concluído concluído com sucesso!')
