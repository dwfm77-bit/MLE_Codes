#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:46:04 2025

@author: usuario
"""

import matplotlib.pyplot as plt
from morph import mm
import cv2, os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage import io, morphology, filters, data, util, img_as_float
from skimage.segmentation import watershed
from skimage.transform import rotate
import rasterio
from skimage.segmentation import find_boundaries
from skimage.morphology import opening, rectangle, square, disk, skeletonize
from skimage.morphology import erosion, reconstruction
from skimage.draw import line as draw_line
from scipy.ndimage import convolve
from skimage.filters import threshold_otsu
from scipy.stats import norm
import tifffile as tiff
import seaborn as sns
import pandas as pd
import itertools
import rasterio
from tqdm import tqdm
from skimage.morphology import disk, dilation, erosion
from skimage.measure import label, regionprops
# from skimage.morphology import remove_small_objects


def iou_np(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-8)


def precision_np(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    return tp / (tp + fp + 1e-8)


def recall_np(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    return tp / (tp + fn + 1e-8)

def skel_iou(mask_ref, mask_pred):
    mask_ref = mask_ref.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.logical_and(mask_ref, mask_pred).sum()
    union = np.logical_or(mask_ref, mask_pred).sum()

    if union == 0:
        return np.nan  # ou 0.0, conforme sua decisão metodológica

    return intersection / union

def skeleton_length_ratio(skel_ref, skel_pred):
    L_ref = np.count_nonzero(skel_ref)
    L_pred = np.count_nonzero(skel_pred)
    if L_ref == 0:
        return np.nan
    return L_pred / L_ref

def skeleton_recall(skel_ref, skel_pred):
    skel_ref = skel_ref.astype(bool)
    skel_pred = skel_pred.astype(bool)
    
    intersection = np.logical_and(skel_ref, skel_pred).sum()
    total_ref = np.count_nonzero(skel_ref)

    if total_ref == 0:
        return np.nan

    return intersection / total_ref

def confusion_elements(pred, ref):
    """
    pred, ref: arrays booleanos (True = estrada)
    """
    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    fn = np.logical_and(~pred, ref).sum()
    tn = np.logical_and(~pred, ~ref).sum()
    return tp, fp, fn, tn

def boundary_f1_score(mask_ref, mask_pred, tolerance=2):
    """
    Boundary F1 Score conforme:
    'What is a good evaluation measure for semantic segmentation?'

    Parameters
    ----------
    mask_ref : np.ndarray (bool ou 0/1)
    mask_pred : np.ndarray (bool ou 0/1)
    tolerance : int
        Tolerância espacial em pixels

    Returns
    -------
    bf_score : float
    """
    # mask_ref  = np.asarray(mask_ref)
    # mask_pred = np.asarray(mask_pred)
    
    # # Normalização de shape
    # if mask_ref.ndim == 3:
    #     mask_ref = mask_ref[0]
    
    # if mask_pred.ndim == 3:
    #     mask_pred = mask_pred[0]
    
    # if mask_ref.ndim != 2 or mask_pred.ndim != 2:
    #     raise ValueError(
    #         f"boundary_f1_score espera máscaras 2D, "
    #         f"recebeu mask_ref={mask_ref.shape}, mask_pred={mask_pred.shape}"
    #     )

    mask_ref = mask_ref.astype(bool)
    mask_pred = mask_pred.astype(bool)

    # Extrai contornos
    boundary_ref = find_boundaries(mask_ref, mode="inner")
    boundary_pred = find_boundaries(mask_pred, mode="inner")

    if boundary_ref.sum() == 0 and boundary_pred.sum() == 0:
        return np.nan

    # Distância até o contorno oposto
    dt_ref = distance_transform_edt(~boundary_ref)
    dt_pred = distance_transform_edt(~boundary_pred)

    # Recall: contornos de referência encontrados na predição
    recall = (dt_pred[boundary_ref] <= tolerance).sum() / max(boundary_ref.sum(), 1)

    # Precision: contornos preditos corretos
    precision = (dt_ref[boundary_pred] <= tolerance).sum() / max(boundary_pred.sum(), 1)

    if precision + recall == 0:
        return 0.0

    bf_score = 2 * precision * recall / (precision + recall)
    return bf_score

def Calc_Metrics(pred, mask_ref):
    
    # --- Classical Metrics Predictions ---
    tp, fp, fn, tn = confusion_elements(pred, mask_ref)
    
    iou = iou_np(pred, mask_ref)
    precision = precision_np(pred, mask_ref)
    recall = recall_np(pred, mask_ref)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
    return iou, precision, recall, f1_score, tp, fp, fn, tn


def extract_metric(metrics_dict, metric_name):
    """
    metrics_dict: {band: {metric: [values]}}
    return: {band: [values]}
    """
    return {
        band: metrics[metric_name]
        for band, metrics in metrics_dict.items()
    }
    
# def extract_metric(metrics_dict, metric_name, output_dir):
    
#     """
#     metrics_dict: {band: {metric: [values]}}
#     return: {band: [values]} """
#     return { band: metrics[metric_name] for band, metrics in metrics_dict.items()}
    
#     extract_metric(metrics_dict, "iou")
#     extract_metric(metrics_dict, "precision")
#     extract_metric(metrics_dict, "recall")
#     extract_metric(metrics_dict, "f1")
#     extract_metric(metrics_dict, "skl_iou")
#     extract_metric(metrics_dict, "skl_f1")
#     extract_metric(metrics_dict, "bf_score")
#     extract_metric(metrics_dict, "median_length_skl")
    
#     df_dilation = metrics_dict_to_dataframe(metrics_dict, "dilation")
#     df_erosion  = metrics_dict_to_dataframe(metrics_dict, "erosion")
    
#     df_dilation.to_csv(
#     os.path.join(output_dir, "metrics_linear_enhancement_dilation.csv"),
#     index=False
#     )
    
#     df_erosion.to_csv(
#         os.path.join(output_dir, "metrics_linear_enhancement_erosion.csv"),
#         index=False
#     )
#=======================================================================================

def skeleton_analysis(pred_s, ste, dilation=False, seg_filter = True, length_percentile=80, min_seg = 5):
    """
    Análise estrutural baseada em esqueletos usando percentis estáveis:
    - calcua as estatísticas relacionadas aos esqueletos gerados
    - processa uma nova imagem:
        1. remove esqueletos curtos com base em percentil fixo
        2. preserva apenas objetos conectados a esqueletos relevantes

    Parameters
    ----------
    pred_s : np.ndarray
        Predição binária ou em níveis de cinza
    ste : np.ndarray
        Elemento estruturante para dilatação
    dilation : bool, optional
        Aplica dilatação antes da esqueletização
    length_percentile : float, optional
        Percentil mínimo de comprimento do esqueleto (0–100)
    """

    # --------------------------------------------------
    # 1. Preparação da predição
    # --------------------------------------------------
    pred_bin = (pred_s > 0).astype(np.uint8)

    if dilation:
        pred_bin = mm.dil(pred_bin, Bc=ste)

    # --------------------------------------------------
    # 2. Esqueletização
    # --------------------------------------------------
    pred_skl = skeletonize(pred_bin, method="zhang").astype(np.uint8)
    
    # --------------------------------------------------
    # 3. Rotulagem dos esqueletos
    # --------------------------------------------------
    skl_labeled = label(pred_skl, connectivity=2)
    skl_props = regionprops(skl_labeled)

    if len(skl_props) == 0:
        return pred_bin, {
            "threshold": None,
            "percentile": length_percentile,
            "n_skeletons": 0
        }

    # --------------------------------------------------
    # 4. Comprimentos e percentil estável
    # --------------------------------------------------
    
    #comprimento dos segmentos
    lengths = np.array([prop.area for prop in skl_props])
    
    if seg_filter:
        min_length = min_seg
        lengths_valid = lengths[lengths >= min_length]
        
        if len(lengths_valid) == 0:
            return pred_bin, {
                "percentile": length_percentile,
                "threshold": None,
                "n_skeletons": int(len(lengths)),
                "n_selected": 0,
                "mean_length": float(lengths.mean()) if len(lengths) > 0 else 0.0,
                "median_length": float(np.median(lengths)) if len(lengths) > 0 else 0.0,
                "std_length": float(lengths.std()) if len(lengths) > 0 else 0.0,
                "min_length": int(lengths.min()) if len(lengths) > 0 else 0,
                "max_length": int(lengths.max()) if len(lengths) > 0 else 0}
            
        
        length_thresh = np.percentile(lengths_valid, length_percentile)#, method="median_unbiased")

        # --------------------------------------------------
        # 5. Seleção dos esqueletos longos
        # --------------------------------------------------
        valid_labels = [
            prop.label for prop in skl_props
            if prop.area >= length_thresh
        ]
    
        skl_selected = np.isin(skl_labeled, valid_labels).astype(np.uint8)
    
        # --------------------------------------------------
        # 6. Seleção dos objetos associados
        # --------------------------------------------------
        pred_labeled = label(pred_bin, connectivity=2)
        pred_selected = np.zeros_like(pred_bin)
    
        for region in regionprops(pred_labeled):
            obj_mask = (pred_labeled == region.label)
    
            if np.any(obj_mask & skl_selected):
                pred_selected[obj_mask] = 1
    
        # --------------------------------------------------
        # 7. Composição final
        # --------------------------------------------------
        img_post_processed = np.clip(
            pred_selected + skl_selected, 0, 1
        ).astype(np.uint8)
        
        # --------------------------------------------------
        # 8. Estatísticas para análise experimental
        # --------------------------------------------------
        pred_skl_stats = {
            "percentile": length_percentile,
            "threshold": length_thresh,
            "n_skeletons": int(len(lengths)),
            "n_selected": int(len(valid_labels)),
            "mean_length": float(lengths.mean()),
            "median_length": float(np.median(lengths)),
            "std_length": float(lengths.std()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max())
        }
        return img_post_processed, pred_skl_stats

    else:
        # --------------------------------------------------
        # Estatísticas para análise experimental
        # --------------------------------------------------
        pred_skl_stats = {
            "percentile": 0,
            "threshold": 0,
            "n_skeletons": int(len(lengths)),
            "n_selected": 0,
            "mean_length": float(lengths.mean()),
            "median_length": float(np.median(lengths)),
            "std_length": float(lengths.std()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max())
        }
        # for stats in pred_skl_stats:
        #     print(f"{stats}: {pred_skl_stats[stats]}")
    
        return pred_skl, pred_skl_stats


def gaussian_gradient(img, sigma=1.5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return normalize(grad)

def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def morphological_gradient(img, radius=2):
    se = disk(radius)
    dil = dilation(img, se)
    ero = erosion(img, se)
    grad = dil - ero
    return normalize(grad)

def otsu_segment(grad):
    t = threshold_otsu(grad)
    return (grad > t).astype(np.uint8)


def compute_metrics(pred, gt):
    """
    pred, gt: arrays binários (0/1) com mesma forma
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)

    return precision, recall, f1, iou

def otsu_balanced(img_norm, mask_gt, max_samples=100000):
    road = img_norm[mask_gt == 1]
    bg   = img_norm[mask_gt == 0]

    n = min(len(road), len(bg), max_samples)

    road_s = np.random.choice(road, n, replace=False)
    bg_s   = np.random.choice(bg, int(n/2), replace=False)

    pixels = np.concatenate([road_s, bg_s])

    return threshold_otsu(pixels)

def save_geotiff_mask(ref_path, out_path, mask):
    """
    Salva máscara binária preservando georreferenciamento.
    """
    with rasterio.open(ref_path) as src:
        meta = src.meta.copy()

    meta.update({
        'count': 1,
        'dtype': rasterio.uint8,
        'nodata': 0
    })

    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(mask.astype(np.uint8), 1)



def top_hat(f, kernel = np.ones((3,3), dtype='uint8'), mode = 'white'):
    if mode == 'white':
        #white top-hat: f - opening(f) This transformation is a very good contrast detector suitable for enhancing the
        #white and narrow objects in the image
        new_img = f - mm.open(f, b=kernel)
    elif mode == 'black':
        #black top-hat: close(f) - f - to enhance black and narrow features
        new_img = mm.close(f, b=kernel) - f
    return new_img

def grad_morph(f, kernel = np.ones((3,3), dtype='uint8')):
    if len(f.shape) != 2:
        f = f[:,:,1]
    try:
        grad = cv2.morphologyEx(f, cv2.MORPH_GRADIENT, kernel)
    except:
        #dilation
        img_dil = mm.dil(f, Bc = kernel)
        #erosion
        img_ero = mm.ero(f, Bc = kernel)
        #grad
        grad = img_dil - img_ero
    
    return grad

def iou_score(pred, target):
    """
    Calcula a métrica Intersection over Union (IoU) entre duas imagens binárias.
    
    Args:
        pred (numpy.ndarray): Imagem binária da predição (valores 0 ou 1).
        target (numpy.ndarray): Imagem binária do ground truth (valores 0 ou 1).
    
    Returns:
        float: Valor de IoU.
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    iou = intersection / union if union != 0 else 0.0
    return iou

def road_seg(img, B = np.ones((3, 3), dtype='uint8')):
    # knl_size = B.shape[0]
    
    # stp = int((knl_size - 1)/2)
    
    # rows, cols = img.shape
  
    # #create a np zero matrix
    # erosion_img = np.zeros((img.shape[0],img.shape[1]))
    
    # #iterate over the img matrix and apply the convolution
    # for i in range(stp, rows - stp):
    #   for j in range(stp, cols - stp):
    #     aux_img = np.dot(img[i - stp:i + stp + 1,j - stp:j + stp + 1].flatten(),B.flatten())
    #     if erosion_img[i,j] == aux_img:
    #         erosion_img[i,j] == aux_img
    
    #img erosion
    erosion_img = mm.ero(img, Bc=B)
    
    print(erosion_img.shape)
    print(img.shape)
    
    # assert the image shapes
    assert erosion_img.shape == img.shape, "As imagens devem ter o mesmo tamanho!"

    # Compare pixels and create a new binary image
    rs_img = (img == erosion_img).astype(np.uint8)

    rs_img = rs_img * 255

    return rs_img 



def Show_S2_imgs_vs1(img_dir, output_dir, img_ops, print_results = False):
    
    image_files = os.listdir(os.path.join(img_dir,'b2'))

    # Definir elemento estruturante
    ste = np.ones((5, 5), dtype='uint8')
    
    # Criar um arquivo PDF para salvar as imagens
    pdf_filename = f"imagens_processadas_{datetime.now().strftime('Sentinel2_Tests_%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    with PdfPages(pdf_path) as pdf:
        for nr, f in enumerate(image_files):
            print(f'Imagem: {nr} de {len(image_files)}')
    
            #Ler as bandas S2
            img_b2 = Image.open(os.path.join(img_dir,'b2', f))
            img_b3 = Image.open(os.path.join(img_dir,'b3', f))
            img_b4 = Image.open(os.path.join(img_dir,'b4', f))
            img_b8 = Image.open(os.path.join(img_dir,'b8', f))
            
            #Converte para 8-bits
            img_b2 = np.array(img_b2.convert("L"))
            img_b3 = np.array(img_b3.convert("L"))
            img_b4 = np.array(img_b4.convert("L"))
            img_b8 = np.array(img_b8.convert("L"))
            
            # img_b2 = mm.read(os.path.join(img_dir,'b2', f))
            # img_b3 = mm.read(os.path.join(img_dir,'b3', f))
            # img_b4 = mm.read(os.path.join(img_dir,'b4', f))
            # img_b8 = mm.read(os.path.join(img_dir,'b8', f))
            
            ##Ler a mascara
            #msk = Image.open(os.path.join(mask_dir, f))
            #msk = (np.array(msk).astype(np.uint8))*255
    
            #Criar subplots
            for i, ops in enumerate(img_ops):
                
                if ops == 'Imagem Original':
                    #imgs to plot
                    imagens = [img_b2, img_b3, img_b4, img_b8]
                    
                elif ops == 'Dilatação':
                    print('Dilatação...')
                    b2_dilation = mm.dil(img_b2, Bc=ste)
                    b3_dilation = mm.dil(img_b3, Bc=ste)
                    b4_dilation = mm.dil(img_b4, Bc=ste)
                    b8_dilation = mm.dil(img_b8, Bc=ste)
                    #imgs to plot
                    imagens = [b2_dilation, b3_dilation, b4_dilation, b8_dilation]
                        
                elif ops == 'Erosão':
                    print('Erosão...')
                    b2_erosion = mm.ero(img_b2, Bc=ste)
                    b3_erosion = mm.ero(img_b3, Bc=ste)
                    b4_erosion = mm.ero(img_b4, Bc=ste)
                    b8_erosion = mm.ero(img_b8, Bc=ste)
                    #imgs to plot
                    imagens = [b2_erosion, b3_erosion, b4_erosion, b8_erosion]
                    
                elif ops == 'Abertura':
                    print('Abertura...')
                    b2_open = mm.open(img_b2, b=ste)
                    b3_open = mm.open(img_b3, b=ste)
                    b4_open = mm.open(img_b4, b=ste)
                    b8_open = mm.open(img_b8, b=ste)
                    #imgs to plot
                    imagens = [b2_open, b3_open, b4_open, b8_open]    
                
                elif ops == 'Fechamento':
                    print('Fechamento...')
                    #Img_close = mm.close(img, b=ste)
                    b2_close = mm.close(img_b2, b=ste)
                    b3_close = mm.close(img_b3, b=ste)
                    b4_close = mm.close(img_b4, b=ste)
                    b8_close = mm.close(img_b8, b=ste)
                    #imgs to plot
                    imagens = [b2_close, b3_close, b4_close, b8_close]
                    
                elif ops == 'White Top-Hat':
                    print('White Top-Hat...')
                    b2_wth = top_hat(img_b2, kernel = ste)
                    b3_wth = top_hat(img_b3, kernel = ste)
                    b4_wth = top_hat(img_b4, kernel = ste)
                    b8_wth = top_hat(img_b8, kernel = ste)
                    #imgs to plot
                    imagens = [b2_wth, b3_wth, b4_wth, b8_wth]

                elif ops == 'Black Top-Hat':
                    print('Black Top-Hat...')
                    b2_bth = top_hat(img_b2, kernel = ste, mode='black')
                    b3_bth = top_hat(img_b3, kernel = ste, mode='black')
                    b4_bth = top_hat(img_b4, kernel = ste, mode='black')
                    b8_bth = top_hat(img_b8, kernel = ste, mode='black')
                    #imgs to plot
                    imagens = [b2_bth, b3_bth, b4_bth, b8_bth]
    
                # Criar figura
                fig = plt.figure(figsize=(20, 12))
                
                # Criar subplots
                for idx, img in enumerate(imagens):
                    ax = plt.subplot(1, 4, idx + 1)
                    ax.set_title(f'Imagem - {ops}')
                    ax.imshow(img, cmap='gray')
                    ax.axis("off")
                
                # # Ajustar layout
                # plt.tight_layout()
                
                if print_results:
                    plt.show()
        
                # Salvar a figura no PDF
                pdf.savefig(fig, dpi=300)  # Salva com 600 dpi
        
                # Fechar a figura para liberar memória
                plt.close(fig)
                
def Show_S1_imgs(img_dir, output_dir, img_ops, print_results=False):
    image_files = os.listdir(os.path.join(img_dir, 'VH'))

    # Definir elemento estruturante
    ste = np.ones((3, 3), dtype='uint8')

    # Criar um arquivo PDF para salvar as imagens
    pdf_filename = f"imagens_processadas_{datetime.now().strftime('Sentinel1_Tests_%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    with PdfPages(pdf_path) as pdf:
        for nr, f in enumerate(image_files):
            print(f'Processando imagem {nr+1}/{len(image_files)}: {f}')

            # Ler as bandas S2 e converter para 8 bits
            img_vh = np.array(Image.open(os.path.join(img_dir, 'VH', f)))
            img_vv = np.array(Image.open(os.path.join(img_dir, 'VV', f)))
            img_alpha = np.array(Image.open(os.path.join(img_dir, 'Alpha', f)))
            img_entropy = np.array(Image.open(os.path.join(img_dir, 'Entropy', f)))

            # Criar lista para armazenar todas as imagens a serem exibidas
            resultados = []

            for ops in img_ops:
                print(f'Aplicando operação: {ops}')

                # Aplicar a operação escolhida
                if ops == 'Imagem Original':
                    imagens = [img_vh, img_vv, img_alpha, img_entropy]
                elif ops == 'Dilatação':
                    imagens = [mm.dil(img_vh, Bc=ste), mm.dil(img_vv, Bc=ste),
                               mm.dil(img_alpha, Bc=ste), mm.dil(img_entropy, Bc=ste)]
                elif ops == 'Erosão':
                    imagens = [mm.ero(img_vh, Bc=ste), mm.ero(img_vv, Bc=ste),
                               mm.ero(img_alpha, Bc=ste), mm.ero(img_entropy, Bc=ste)]
                elif ops == 'Abertura':
                    imagens = [mm.open(img_vh, b=ste), mm.open(img_vv, b=ste),
                               mm.open(img_alpha, b=ste), mm.open(img_entropy, b=ste)]
                elif ops == 'Fechamento':
                    imagens = [grad_morph(img_vh, kernel=ste), grad_morph(img_vv, kernel=ste),
                               grad_morph(img_alpha, kernel=ste), grad_morph(img_entropy, kernel=ste)]
                elif ops == 'White Top-Hat':
                    imagens = [top_hat(img_vh, kernel=ste), top_hat(img_vv, kernel=ste),
                               top_hat(img_alpha, kernel=ste), top_hat(img_entropy, kernel=ste)]
                elif ops == 'Black Top-Hat':
                    imagens = [top_hat(img_vh, kernel=ste, mode='black'), top_hat(img_vv, kernel=ste, mode='black'),
                               top_hat(img_alpha, kernel=ste, mode='black'), top_hat(img_entropy, kernel=ste, mode='black')]
                else:
                    print(f"Operação {ops} não reconhecida. Pulando...")
                    continue

                resultados.append((ops, imagens))  # Armazena os resultados para exibição

            # Criar a figura para essa imagem
            num_operacoes = len(resultados)
            fig, axs = plt.subplots(num_operacoes, 4, figsize=(20, 5 * num_operacoes))

            # Criar os subplots
            bandas = ['VH', 'VV', 'Alpha', 'Entropy']
            for row_idx, (operacao, imagens) in enumerate(resultados):
                for col_idx, (ax, img, banda) in enumerate(zip(axs[row_idx], imagens, bandas)):
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"{banda}\n{operacao}", fontsize=10)
                    ax.axis("off")

            # Ajustar layout e exibir se necessário
            plt.tight_layout()
            if print_results:
                plt.show()

            # Salvar a figura no PDF
            pdf.savefig(fig, dpi=300)
            plt.close(fig)  # Fechar a figura para liberar memória

#exibe a aplicacao de filtros sobre todas as bandas sentinel-2
def Show_S2_imgs(img_dir, output_dir, img_ops, band_list, print_results=False):
    image_files = os.listdir(os.path.join(img_dir, 'b2'))

    # Definir elemento estruturante
    ste = np.ones((3, 3), dtype='uint8')

    # Criar um arquivo PDF para salvar as imagens
    pdf_filename = f"imagens_processadas_{datetime.now().strftime('Sentinel1_Tests_%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    with PdfPages(pdf_path) as pdf:
        for nr, f in enumerate(image_files):
            print(f'Processando imagem {nr+1}/{len(image_files)}: {f}')

            # Ler as bandas S2 e converter para 8 bits
            try:
                img_b1 = np.array(Image.open(os.path.join(img_dir, band_list[0], f)).convert("L"))
            except:
                with rasterio.open(os.path.join(img_dir, band_list[0], f)) as src:
                    img_b1 = src.read(1)
            try:
                img_b2 = np.array(Image.open(os.path.join(img_dir, band_list[1], f)).convert("L"))
            except:
                with rasterio.open(os.path.join(img_dir, band_list[1], f)) as src:
                    img_b2 = src.read(1)
            try:
                img_b3 = np.array(Image.open(os.path.join(img_dir, band_list[2], f)).convert("L"))
            except:
                with rasterio.open(os.path.join(img_dir, band_list[2], f)) as src:
                    img_b3 = src.read(1)
            try:
                img_b4 = np.array(Image.open(os.path.join(img_dir, band_list[3], f)).convert("L"))
            except:
                with rasterio.open(os.path.join(img_dir, band_list[3], f)) as src:
                    img_b4 = src.read(1)
            try:
                img_b5 = np.array(Image.open(os.path.join(img_dir, band_list[4], f)).convert("L"))
            except:
                with rasterio.open(os.path.join(img_dir, band_list[4], f)) as src:
                    img_b5 = src.read(1)

            # Criar lista para armazenar todas as imagens a serem exibidas
            resultados = []

            for ops in img_ops:
                print(f'Aplicando operação: {ops}')

                # Aplicar a operação escolhida
                if ops == 'Imagem Original':
                    imagens = [img_b1, img_b2, img_b3, img_b4, img_b5]
                elif ops == 'Dilatação':
                    imagens = [mm.dil(img_b1, Bc=ste), mm.dil(img_b2, Bc=ste),
                               mm.dil(img_b3, Bc=ste), mm.dil(img_b4, Bc=ste), mm.dil(img_b5, Bc=ste)]
                elif ops == 'Erosão':
                    imagens = [mm.ero(img_b1, Bc=ste), mm.ero(img_b2, Bc=ste),
                               mm.ero(img_b3, Bc=ste), mm.ero(img_b4, Bc=ste), mm.ero(img_b5, Bc=ste)]
                elif ops == 'Abertura':
                    imagens = [mm.open(img_b1, b=ste), mm.open(img_b2, b=ste),
                               mm.open(img_b3, b=ste), mm.open(img_b4, b=ste), mm.open(img_b5, b=ste)]
                elif ops == 'Fechamento':
                    imagens = [mm.close(img_b1, b=ste), mm.close(img_b2, b=ste),
                               mm.close(img_b3, b=ste), mm.close(img_b4, b=ste), mm.close(img_b5, b=ste)]
                elif ops == 'White Top-Hat':
                    imagens = [top_hat(img_b1, kernel=ste), top_hat(img_b2, kernel=ste),
                               top_hat(img_b3, kernel=ste), top_hat(img_b4, kernel=ste), top_hat(img_b5, kernel=ste)]
                elif ops == 'Black Top-Hat':
                    imagens = [top_hat(img_b1, kernel=ste, mode='black'), top_hat(img_b2, kernel=ste, mode='black'),
                               top_hat(img_b3, kernel=ste, mode='black'), top_hat(img_b4, kernel=ste, mode='black'),
                               top_hat(img_b5, kernel=ste, mode='black')]
                else:
                    print(f"Operação {ops} não reconhecida. Pulando...")
                    continue

                resultados.append((ops, imagens))  # Armazena os resultados para exibição

            # Criar a figura para essa imagem
            num_operacoes = len(resultados)
            fig, axs = plt.subplots(num_operacoes, 5, figsize=(20, 5 * num_operacoes))

            # Criar os subplots
            #bandas = ['B2 (Azul)', 'B3 (Verde)', 'B4 (Vermelho)', 'B8 (Infravermelho)', 'NDBI']
            for row_idx, (operacao, imagens) in enumerate(resultados):
                for col_idx, (ax, img, banda) in enumerate(zip(axs[row_idx], imagens, band_list)):
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"{banda}\n{operacao}", fontsize=10)
                    ax.axis("off")

            # Ajustar layout e exibir se necessário
            plt.tight_layout()
            if print_results:
                plt.show()

            # Salvar a figura no PDF
            pdf.savefig(fig, dpi=300)
            plt.close(fig)  # Fechar a figura para liberar memória


def watershed_vs0(img):
    
    
    if len(img.shape) == 2:  # Se for imagem em tons de cinza
        img = cv2.merge([img, img, img])

    # Aplicar uma suavização para reduzir ruído
    blur = cv2.GaussianBlur(img, (5,5), 0)
    
    blur = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Obter o mapa de gradientes com a detecção de bordas
    edges = cv2.Canny(blur, 50, 150)
    
    # Transformação morfológica para fechar pequenos buracos
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Encontrar marcadores (background e foreground)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    fg = np.uint8(fg)
    
    # Encontrar a região de fundo
    bg = cv2.dilate(closing, kernel, iterations=3)
    unknown = cv2.subtract(bg, fg)
    
    # Criar marcadores para o watershed
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1  # Evitar conflitos com a região de fundo
    markers[unknown == 255] = 0  # Região desconhecida
    
    # Aplicar o algoritmo watershed
    cv2.watershed(blur, markers)
    
    # Marcar as bordas em vermelho
    img[markers == -1] = [0, 0, 255]
    
    # Exibir resultado
    plt.imshow(img, cmap="gray")
    plt.axis("off")  # Oculta os eixos
    plt.show()
    
def watershed_vs1(img, img_marker, kernel_size=3, return_vis=False):

    # --- RELEVO ---
    img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # blur = cv2.GaussianBlur(img_8u, (5, 5), 0)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    
    img_8u = mm.ero(img_8u, Bc=kernel)
    
    img_color = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)

    # --- MARKERS ---
    #marker_aux = skeletonize(img_marker > 0)
    marker_aux = (img_marker > 0).astype(np.uint8)

    # FORÇAR 2D (ponto crítico)
    if marker_aux.ndim == 3:
        marker_aux = marker_aux[:, :, 0]

    if np.count_nonzero(marker_aux) == 0:
        raise ValueError("Imagem de marcadores vazia após esqueletização.")

    _, markers = cv2.connectedComponents(marker_aux)
    markers = markers.astype(np.int32)

    # --- WATERSHED ---
    cv2.watershed(img_color, markers)

    return markers, img_color



def create_line_se(length, angle):
    """Cria um elemento estruturante do tipo linha (agulha) rotacionado."""
    size = length
    se = np.zeros((size, size), dtype=bool)
    rr, cc = draw_line(size // 2, 0, size // 2, size - 1)
    se[rr, cc] = 1
    rotated = rotate(se.astype(float), angle, order=0, preserve_range=True).astype(bool)
    return rotated

def create_custom_needle(length=21, thickness=1, angle=0, antialiased=True):
    """
    Cria um elemento estruturante do tipo 'agulha' (linha com espessura e rotação controladas).
    
    Parâmetros:
        length : int
            Comprimento da agulha (em pixels).
        thickness : int
            Espessura (largura transversal à linha), em pixels.
        angle : float
            Ângulo da linha em graus (0 = horizontal).
        antialiased : bool
            Se True, retorna uma máscara suavizada (float); senão, retorna bool.
    
    Retorna:
        se : ndarray
            Elemento estruturante 2D.
    """
    size = length + 2 * thickness  # padding para não cortar bordas
    canvas = np.zeros((size, size), dtype=float)
    
    # Ponto central
    cy, cx = size // 2, size // 2
    half_len = length // 2

    # Linha horizontal no centro
    rr, cc = line(cy, cx - half_len, cy, cx + half_len)
    canvas[rr, cc] = 1.0

    # Convolui com um disco para espessura
    kernel = disk(thickness)
    canvas = convolve(canvas, kernel, mode='constant', cval=0.0)

    # Normaliza para 0–1
    canvas /= canvas.max()

    # Rotaciona a linha para o ângulo desejado
    rotated = rotate(canvas, angle, order=1, preserve_range=True)

    # Retorna SE com ou sem anti-aliasing
    return rotated if antialiased else (rotated > 0.2)


def enhance_linear_structures(image, length=3, angle_step= 45, show_results = False):
    """
    Realça estruturas lineares (como estradas) usando abertura morfológica
    com elementos estruturantes em forma de agulha (segmentos de reta) em múltiplas orientações.
    
    Parâmetros:
        image : ndarray
            Imagem em níveis de cinza (2D).
        length : int
            Comprimento da agulha (elemento estruturante linear). Use 3 para estradas de 2-3 pixels.
        angle_step : int
            Intervalo angular entre agulhas (em graus), ex: 15° → 12 orientações.

    Retorna:
        result : ndarray
            Imagem realçada, com resposta máxima entre as aberturas direcionais.
    """
    orientations = np.arange(0, 180, angle_step)
    responses = []

    for angle in orientations:
        line_se = create_line_se(length, angle)
        line_se = line_se.astype('uint8')
        #print(line_se)
        opened = mm.open(image, b = line_se)
        
        if show_results:
            # Visualizar resultado
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("Elemento Estruturante")
            plt.imshow(line_se, cmap="gray")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.title(f"Angulo: {angle}")
            plt.imshow(opened, cmap="gray")
            plt.axis("off")
            plt.show()
        
        responses.append(opened)

    result = np.maximum.reduce(responses)
    return result

def filter_linear_structures(image, length=3, angle_step= 45, show_results = False):

    orientations = np.arange(0, 180, angle_step)
    responses = []

    for angle in orientations:
        line_se = create_line_se(length, angle)
        line_se = line_se.astype('uint8')
        #print(line_se)
        opened = mm.open(image, b = line_se)
        
        if show_results:
            # Visualizar resultado
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("Elemento Estruturante")
            plt.imshow(line_se, cmap="gray")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.title(f"Angulo: {angle}")
            plt.imshow(opened, cmap="gray")
            plt.axis("off")
            plt.show()
        
        responses.append(opened)

    result = np.maximum.reduce(responses)
    return result


def RSbyWatershed(img, mask, ste = np.ones((3, 3), dtype='uint8')):
    
    # #erosão
    img_ero = mm.ero(img, Bc=ste)
    
    #calcula o gradiente morfologico
    gm_img = grad_morph(img_ero,kernel=ste)
    
    # #Calcula o watershed
    # markers, img_vis = watershed_vs1(img, mask, return_vis=True)
    watershed_vs0(gm_img)
    
    
    return img_ero, gm_img #markers, img_vis
    
                
def S2_watershed_test(img_dir, output_dir, lista_bandas, print_results = False, ste = np.ones((3, 3), dtype='uint8')):
    

    # Criar um arquivo PDF para salvar as imagens
    pdf_filename = f"imagens_processadas_{datetime.now().strftime('Sentinel2_Watershed_Tests_%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    lista_imagens = os.listdir(os.path.join(img_dir, 'b2'))
    
    with PdfPages(pdf_path) as pdf:
        for nr, f in enumerate(lista_imagens):
            print(f'Processando imagem {nr+1}/{len(lista_imagens)}: {f}')
            resultados = []
            for banda in lista_bandas:
                #Abre a imagem
    
                try:
                    if banda == 'ndvi' or banda == 'ndwi' or banda == 'ndbi':
                        img_s2 = np.array(Image.open(os.path.join(img_dir, banda, f)))
                    else:
                        img_s2 = np.array(Image.open(os.path.join(img_dir, banda, f)).convert("L"))
                    
                except:
                    with rasterio.open(os.path.join(img_dir, banda, f)) as src:
                        img_s2 = src.read(1)
                        
                #Abre a mascara
                msk_s2 = np.array(Image.open(os.path.join(img_dir, 'Mask', f)).convert("L"))
                
                #calcula o gradiente morfologico
                gm_img = grad_morph(img_s2,kernel=ste)
                
                #Calcula o watershed
                w_img, w_markers = watershed_vs1(img_s2, msk_s2, banda)
                
                
                imagens = [img_s2, msk_s2, w_markers, w_img, gm_img]
                
                resultados.append((banda, imagens))  # Armazena os resultados para exibição
    
            # Criar a figura para essa imagem
            num_operacoes = len(resultados)
            fig, axs = plt.subplots(num_operacoes, 5, figsize=(20, 5 * num_operacoes))
    
            # Criar os subplots
            #bandas = ['B2 (Azul)', 'B3 (Verde)', 'B4 (Vermelho)', 'B8 (Infravermelho)', 'NDBI']
            for row_idx, (banda, imagens) in enumerate(resultados):
                #for col_idx, (ax, img, banda) in enumerate(zip(axs[row_idx], imagens, lista_bandas)):
      
                # Plota os resultados
                axs[row_idx, 0].imshow(imagens[0], cmap='gray')
                axs[row_idx, 0].set_title(f'{banda.upper()} - Original')
                axs[row_idx, 1].imshow(imagens[1], cmap='gray')
                axs[row_idx, 1].set_title(f'{banda.upper()} - Máscara')
                axs[row_idx, 2].imshow(imagens[2], cmap='nipy_spectral')
                axs[row_idx, 2].set_title(f'{banda.upper()} - Markers')
                axs[row_idx, 3].imshow(imagens[3], cmap='gray')
                axs[row_idx, 3].set_title(f'{banda.upper()} - Watershed')
                axs[row_idx, 4].imshow(imagens[4], cmap='gray')
                axs[row_idx, 4].set_title(f'{banda.upper()} - Gradiente')
    
            # Ajustar layout e exibir se necessário
            plt.tight_layout()
            if print_results:
                plt.show()
        
            # Salvar a figura no PDF
            pdf.savefig(fig, dpi=300)
            plt.close(fig)  # Fechar a figura para liberar memória
            

def filtrar_feats_lineares(mask_bin, min_length=5, min_ratio=3.0):
    """
    Filtra feições conectadas mantendo apenas as mais lineares.
    
    Parâmetros:
    - mask_bin: imagem binária (bool ou 0/1)
    - min_length: comprimento mínimo da feição
    - min_ratio: razão entre eixo maior / menor
    
    Retorna:
    - máscara com feições lineares preservadas
    """
    label_img = label(mask_bin)
    output = np.zeros_like(mask_bin, dtype=bool)
    
    for region in regionprops(label_img):
        if region.area < 5:
            continue  # ignora ruído pequeno
        
        # Propriedades geométricas
        major = region.major_axis_length
        minor = region.minor_axis_length if region.minor_axis_length > 0 else 1
        
        elongation = major / minor
        if major >= min_length and elongation >= min_ratio:
            output[label_img == region.label] = True
    
    return output

def segmentador_ci(imgs_dir, img, ic_por_banda, img_nr, show_results = True):
    """ segmentação de imagens a partir da seleção de pixels dentro de um intervalo de confiança"""    
    
    b2_img = np.array(Image.open(os.path.join(imgs_dir, 'b2', img)).convert("L"))
    road_pixels = np.ones(b2_img.shape, dtype=bool)
        
    for b in bandas:
        try:
            if b == 'ndvi' or b == 'ndwi' or b == 'ndbi':
                b_img = np.array(Image.open(os.path.join(imgs_dir, b, img)))
            else:
                b_img = np.array(Image.open(os.path.join(imgs_dir, b, img)).convert("L"))
            
        except:
            with rasterio.open(os.path.join(imgs_dir, b, img)) as src:
                b_img = src.read(1)
    
        if b_img.shape != b2_img.shape:
            b_img = cv2.resize(b_img, (b2_img.shape[1], b2_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        if len(b_img.shape) > 2:
            b_img = b_img[:, :, 0]

        
        ic = ic_por_banda[b]
        ic_interval = np.logical_and(b_img >= ic[0], b_img <= ic[1])
        road_pixels = np.logical_and(road_pixels, ic_interval)
    
    
    # 4. Gerar imagem final com pixels que atendem a todos os ICs
    imagem_filtrada = np.where(road_pixels, 255, 0).astype(np.uint8)
    #imagem_filtrada = np.where(road_pixels, 0, 255).astype(np.uint8)
    
    # Realçar estradas
    enhanced = enhance_linear_structures(imagem_filtrada, length=3, angle_step=45)
    #enhanced2 = skeletonize(imagem_filtrada)
    #enhanced3 = filtrar_feats_lineares(imagem_filtrada, min_length=5, min_ratio=3.5)
    
    if show_results:
        # Visualizar resultado
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Imagem {img_nr} Filtrada")
        plt.imshow(imagem_filtrada, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.title(f"Imagem {img_nr} Realçada")
        plt.imshow(enhanced, cmap="gray")
        plt.axis("off")
        plt.show()

def estimador_ci(img_dir, mask_dir, bandas, IC):
    """estima o intervalo de confiança para um conjunto de imagens
    bandas: lista com as bandas,
    IC: intervalo de confiança desejado"""
    
    # Dicionário para armazenar pixels de estrada por banda
    road_pixels_por_banda = {b: [] for b in bandas}
    images_dir = os.path.join(s2_dir, bandas[0])
    all_road_pixels = []
    shape = 0

    for img in os.listdir(images_dir):                            
        mask = np.array(Image.open(os.path.join(mask_dir, img)).convert("L"))
        b2_img = np.array(Image.open(os.path.join(img_dir, 'b2', img)).convert("L"))
        if shape == 0:
            shape = b2_img.shape
        
        if mask.shape != b2_img.shape[:2]:
            # print("Ajustando marcadores para o tamanho da imagem...")
            mask = cv2.resize(mask, (b2_img.shape[1], b2_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask_aux = skeletonize(mask > 0)
        
        for b in bandas:
            try:
                if b == 'ndvi' or b == 'ndwi' or b == 'ndbi':
                    b_img = np.array(Image.open(os.path.join(img_dir, b, img)))
                else:
                    b_img = np.array(Image.open(os.path.join(img_dir, b, img)).convert("L"))
                
            except:
                with rasterio.open(os.path.join(img_dir, b, img)) as src:
                    b_img = src.read(1)
            
            #reduz o nr de canais a 1
            if len(b_img.shape) > 2:
                b_img = b_img[:, :, 0]
    
            #redimensiona as imagens se necessarios
            if b2_img.shape != b_img.shape[:2]:
                b_img = cv2.resize(b_img, (b2_img.shape[1], b2_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
            pixels_estrada = b_img[mask_aux == 1]
            road_pixels_por_banda[b].extend(pixels_estrada.tolist())
            
        # #road_mean = b2_img[mask_aux == 1].mean()
       
        # #seleciona os pixels das estradas
        # road_img = np.where(mask_aux == 1, b2_img, 0)
          
        # #otsu_thresh = threshold_otsu(image_band)
    
        # # Extrai os valores da imagem onde a máscara indica estrada
        # road_pixels = b2_img[mask_aux == 1]
        # all_road_pixels.extend(road_pixels.tolist())
        
    # 2. Calcular intervalo de confiança 95% por banda
    ic_por_banda = {}
    for b in bandas:
        dados = np.array(road_pixels_por_banda[b])
        mu = np.mean(dados)
        sigma = np.std(dados)
        ic = norm.interval(IC, loc=mu, scale=sigma)
        ic_por_banda[b] = ic
        #print(f"{b}: IC95% = ({ic[0]:.2f}, {ic[1]:.2f})")
    return ic_por_banda


def img_compare(img, tp, B = np.ones((3, 3), dtype='uint8')):
    ''' tp = type of: 'erosion' or 'dilation' '''
    if tp == 'erosion':
        #img erosion
        m_img = cv2.erode(img, B, iterations=1)
    if tp == 'dilation':
        m_img = cv2.dilate(img, B, iterations=1)
    
    # assert the image shapes
    if m_img.shape != img.shape:
        assert m_img.shape == img.shape, "The images must have the same size!"

    # Compare pixels and create a new binary image
    rs_img = (img == m_img).astype(np.uint8)

    rs_img = rs_img * 255

    return rs_img

# def create_line_se(length, angle):
#     """Creates a structuring element of the rotated line (needle) type."""
#     size = length
#     se = np.zeros((size, size), dtype=bool)
#     rr, cc = draw_line(size // 2, 0, size // 2, size - 1)
#     se[rr, cc] = 1
#     rotated = rotate(se.astype(float), angle, order=0, preserve_range=True).astype(bool)
#     return rotated

def create_thick_line_se(length, angle, thickness=2):
    """
    Cria um elemento estruturante linear rotacionado
    com espessura controlada (>=1).
    """

    size = length
    se = np.zeros((size, size), dtype=bool)

    # linha central
    rr, cc = draw_line(size // 2, 0, size // 2, size - 1)
    se[rr, cc] = True

    # espessamento perpendicular
    if thickness > 1:
        perp_se = np.ones((thickness, thickness), dtype=bool)
        se = binary_dilation(se, structure=perp_se)

    # rotação
    se_rot = rotate(
        se.astype(float),
        angle,
        order=0,
        preserve_range=True
    ).astype(bool)

    return se_rot.astype(np.uint8)

def directional_response(prob_img, se):
    """
    Soma das probabilidades sob o elemento estruturante.
    """
    se = se.astype(np.float32)
    return cv2.filter2D(prob_img, -1, se, borderType=cv2.BORDER_REFLECT)

##
def linear_probabilistic_dilation(
    prob_img,
    length,
    thickness=2,
    directions=(0, 45, 90, 135),
    normalize=True
):
    """
    Dilatação linear orientada baseada na soma de probabilidades.
    """

    # cria SEs
    ses = {
        ang: create_thick_line_se(length, ang, thickness)
        for ang in directions
    }

    # respostas direcionais
    responses = []
    for ang in directions:
        r = directional_response(prob_img, ses[ang])
        responses.append(r)

    responses = np.stack(responses, axis=0)  # (ndirs, H, W)

    # direção vencedora por pixel
    best_dir = np.argmax(responses, axis=0)

    # dilatações clássicas por direção
    dilated = []
    for ang in directions:
        d = cv2.dilate(prob_img, ses[ang])
        dilated.append(d)

    dilated = np.stack(dilated, axis=0)

    # seleção pixel-wise da dilatação vencedora
    out = np.zeros_like(prob_img, dtype=np.float32)
    for i in range(len(directions)):
        out[best_dir == i] = dilated[i][best_dir == i]

    if normalize:
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)

    return out


def linear_enhance(img, length, operation, show_imgs=False, thickness=1):

    h_line  = create_thick_line_se(length, 0,   thickness)
    ne_line = create_thick_line_se(length, 45,  thickness)
    v_line  = create_thick_line_se(length, 90,  thickness)
    no_line = create_thick_line_se(length, 135, thickness)

    h_enhanced  = img_compare(img, operation, B=h_line)
    ne_enhanced = img_compare(img, operation, B=ne_line)
    v_enhanced  = img_compare(img, operation, B=v_line)
    no_enhanced = img_compare(img, operation, B=no_line)

    img_c = (h_enhanced | v_enhanced | ne_enhanced | no_enhanced).astype(np.uint8)

    if show_imgs:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 4, 1)
        plt.title("Linear_0°")
        plt.imshow(h_enhanced, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 4, 2)
        plt.title("Linear_45°")
        plt.imshow(ne_enhanced, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 4, 3)
        plt.title("Linear_90°")
        plt.imshow(v_enhanced, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 4, 4)
        plt.title("Linear_135°")
        plt.imshow(no_enhanced, cmap="gray")
        plt.axis("off")
        
        plt.show()

        plt.imshow(img_c, cmap='gray')
        #plt.title("Composed")
        plt.axis("off")
        plt.show()
        
    return img_c

    

def plot_metrics_all_bands(
    metrics_dict,
    metric_name,
    title=None,
    ylabel=None,
    figsize=(10, 6),
    show_stats=True
):
    # --- construir dataframe em formato longo ---
    rows = []
    for band, values in metrics_dict.items():
        for v in values:
            if v is not None and not np.isnan(v):
                rows.append({
                    "Band": band,
                    metric_name: v
                })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[AVISO] DataFrame vazio para métrica {metric_name}")
        return

    # --- boxplot ---
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="Band",
        y=metric_name,
        showfliers=True
    )

    plt.xlabel("Band")
    plt.ylabel(ylabel if ylabel else metric_name)
    plt.title(title if title else metric_name)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- estatísticas opcionais ---
    if show_stats:
        stats = (
            df.groupby("Band")[metric_name]
              .agg(["mean", "median", "std", "min", "max"])
        )
        print(f"\nResumo estatístico – {metric_name}")
        print(stats)
        
        
#old version of linear enhance funtion
# def linear_enhance(img, length, operation, show_imgs = False):
#     '''applies conditional dilation operations with linear structuring element in 4 directions (0°, 45°, 90°, 135°)'''

#     #LW filter
#     h_line = create_line_se(length, 0)
#     h_line = h_line.astype(np.uint8)
#     h_enhanced = img_compare(img, operation, B = h_line)
        
#     #NE filter
#     ne_line = create_line_se(length, 45)
#     ne_line = ne_line.astype(np.uint8)
#     ne_enhanced = img_compare(img, operation, B = ne_line)
    
#     #NS filter
#     v_line = create_line_se(length, 90)
#     v_line = v_line.astype(np.uint8)
#     v_enhanced = img_compare(img, operation, B = v_line)
    
#     #NO filter
#     no_line = create_line_se(length, 135)
#     no_line = no_line.astype(np.uint8)
#     no_enhanced = img_compare(img, operation, B = no_line)
    
#     #Image composition
#     img_c = (h_enhanced | v_enhanced | ne_enhanced | no_enhanced).astype(np.uint8)
    
#     if show_imgs:    
#         plt.figure(figsize=(12, 5))
#         plt.subplot(1, 4, 1)
#         plt.title("Linear_0°")
#         plt.imshow(h_enhanced, cmap="gray")
#         plt.axis("off")
        
#         plt.subplot(1, 4, 2)
#         plt.title("Linear_45°")
#         plt.imshow(ne_enhanced, cmap="gray")
#         plt.axis("off")
        
#         plt.subplot(1, 4, 3)
#         plt.title("Linear_90°")
#         plt.imshow(v_enhanced, cmap="gray")
#         plt.axis("off")
        
#         plt.subplot(1, 4, 4)
#         plt.title("Linear_135°")
#         plt.imshow(no_enhanced, cmap="gray")
#         plt.axis("off")
        
#         plt.show()

#         plt.imshow(img_c)
#         plt.title("Composed")
#         plt.axis("off")
#         plt.show()
        
#     return img_c

    
def obr(image, B):
    '''Performs an opening by reconstruction'''
    # 1. Erosion in original image
    e_img = cv2.erode(image, B, iterations=1)
    
    # Normaliza para [0,1]
    image_f = img_as_float(image/255.0)
    e_img_f = img_as_float(e_img/255.0)
    
    plt.imshow(e_img_f)
    plt.title('Eroded Img')
    plt.axis("off")
    plt.show()  
    
    # 2. Morphological reconstruction by conditional dilation
    opened_by_reconstruction = reconstruction(e_img_f, image_f, method = 'dilation')
    
    return opened_by_reconstruction

def calc_metricas(pred, img_mask):
    
    gt = img_mask.astype(np.uint8)
                                    
    # assert the image shapes
    if pred.shape != gt.shape:
        #assert pred.shape == gt.shape, "As imagens devem ter o mesmo tamanho!"
        gtr = cv2.resize(gt, (pred.shape[1],pred.shape[0]))
    else:
        gtr = gt    

    TP = np.logical_and(pred == 255, gtr == 1).sum()
    TN = np.logical_and(pred == 0, gtr == 0).sum()
    FP = np.logical_and(pred == 255, gtr == 0).sum()
    FN = np.logical_and(pred == 0, gtr == 1).sum()
    
    # Calcula métricas
    #Accuracy
    acc_aux = (TP + TN) / (TP + TN + FP + FN)
    accuracy = acc_aux*100
    #Precision
    prec_aux = (TP / (TP + FP) if TP + FP > 0 else 0)
    precision = prec_aux*100
    #F1
    re_aux = (TP / (TP + FN) if TP + FN > 0 else 0)
    recall = re_aux*100
    f1 = (2 * prec_aux * re_aux / (prec_aux + re_aux) if prec_aux + re_aux > 0 else 0)
    #IoU
    iou_aux = (TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0)
    iou = iou_aux*100
    
    return accuracy, precision, f1, iou

def F1_Score(pred, img_mask):

    pred = pred.astype(bool)
    gt   = img_mask.astype(bool)

    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.uint8),
                        (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (2 * precision * recall /
          (precision + recall)) if (precision + recall) > 0 else 0

    return f1# * 100

def IOU(pred, img_mask):

    pred = pred.astype(bool)
    gt   = img_mask.astype(bool)

    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.uint8),
                        (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()

    iou = intersection / union if union > 0 else 0

    return iou# * 100


def plot_metric_per_band(metric_array, band_list, ylabel, color='blue'):
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_per_band = metric_array.mean(axis=0)
    
    bars = ax.bar(band_list, mean_per_band, color=color)
    ax.set_title(f'{ylabel} por banda')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Adiciona os valores nas barras
    ax.bar_label(bars, fmt='%.1f', padding=3)

    plt.tight_layout()
    return fig

def save_metric_boxplot(data, metric, title, filename, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='Banda', y=metric, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Banda')
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(base_output_path, filename), dpi=300)
    plt.close(fig)

def save_metric_barplot(df, metric, title, filename, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Banda', y=metric, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Banda')
    if ylabel:
        ax.set_ylabel(ylabel)
    for i, row in df.iterrows():
        ax.text(i, row[metric] + 0.01, f"{row[metric]:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    fig.savefig(os.path.join(base_output_path, filename), dpi=300)
    plt.close(fig)
    

#Apply and displays the filters over S2 bands and export one pdf file
def Road_Enhance_S2_imgs(img_dir, output_dir, band_list, operacao, length):
    image_files = os.listdir(os.path.join(img_dir, 'b2'))
    image_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff'))]

    # Definir elemento estruturante
    ste = np.ones((5, 5), dtype='uint8')
    nr_imagens_pg = 5
    nr_imagens_banda = 20

    #Dicionario de metricas
    iou_hist = {b: [] for b in band_list}
    f1_hist = {b: [] for b in band_list}
    
    obr_iou_hist = {b: [] for b in band_list}
    obr_f1_hist = {b: [] for b in band_list}
    
    # Criar um arquivo PDF para salvar as imagens
    pdf_filename = f"Realce_linear_{operacao}_{datetime.now().strftime('Sentinel2_Tests_%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    with PdfPages(pdf_path) as pdf:
        rl_set = []
        rl_set_obr = []
        
        for b in band_list:
            imagens = []
            masks = []
            
            print(f'Processando banda: {b}')
            
            # image_files = os.listdir(os.path.join(img_dir, b))  # <-- corrigido aqui
            # image_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff'))]
                        
            for nr, f in enumerate(image_files):

                #Abre as mascaras
                img_mask =  np.array(Image.open(os.path.join(img_dir, 'Mask', f)).convert("L"))
                img_mask = skeletonize(img_mask)
                masks.append(img_mask)
                
                #Importa as imagens
                img_s = tiff.imread(os.path.join(img_dir, b, f))
                if len(img_s.shape) > 2:
                    img_s = img_s[:,:,0]
                
                imagens.append(img_s)
                             
                #suaviza a imagem
                #img_suavizada = cv2.GaussianBlur(img_s2, (5, 5), sigmaX=1)
                #img_s = cv2.blur(img_s, (5, 5))
                #img_suavizada = cv2.medianBlur(img_s2, 5)
                #img_s = img_obr
              
                #realce linear por erosao da banda original
                img_enhanced =  linear_enhance(img_s, length, operacao)
                rl_set.append(img_enhanced)
                
                #calculo dos indicadores
                #accuracy_index, precision_index, f1_index, iou_index = calc_metricas(img_enhanced, img_mask)
                iou_index = IOU(img_enhanced, img_mask)
                f1_index = F1_Score(img_enhanced, img_mask)
              
                #salva indicadores em dicionarios
                iou_hist[b].append(iou_index)
                f1_hist[b].append(f1_index)
                
                #realce linear por erosao da imagem processada por OBR
                
                #aplica a operacao de abertura por reconstrucao
                img_obr = obr(img_s, ste)
                
                img_enhanced_obr =  linear_enhance(img_obr, length, operacao)
                rl_set_obr.append(img_enhanced_obr)

                # #imprime a imagem
                # plt.figure(figsize=(12, 5))
                # plt.subplot(1, 3, 1)
                # plt.title(f"Original {f} - banda {b}")
                # plt.imshow(img_s, cmap="gray")
                # plt.axis("off")
                
                # plt.subplot(1, 3, 2)
                # plt.title("Imagem OBR_vs1")
                # plt.imshow(img_obr, cmap="gray")
                # plt.axis("off")
                
                # plt.subplot(1, 3, 3)
                # plt.title("Imagem Realce Linear_vs1")
                # plt.imshow(img_enhanced_obr, cmap="gray")
                # plt.axis("off")
                # plt.show()
             
                #calculo dos indicadores
                #accuracy_index, precision_index, f1_index, iou_index = calc_metricas(img_enhanced, img_mask)
                iou_index = IOU(img_enhanced_obr, img_mask)
                f1_index = F1_Score(img_enhanced_obr, img_mask)
                
                #salva indicadores em dicionarios
                obr_iou_hist[b].append(iou_index)
                obr_f1_hist[b].append(f1_index)
              
                ################################################################                        
                # Criar a figura para essa imagem
                if nr <= nr_imagens_banda:
                    if len(imagens) == nr_imagens_pg or nr == (len(image_files) - 1):
            
                        fig, axs = plt.subplots(len(imagens), 4, figsize=(20, 15))
                        
                        # Criar os subplots
                        for row_idx, img in enumerate(imagens):
                            
                            axs[row_idx, 0].imshow(imagens[row_idx], cmap='gray')
                            axs[row_idx, 0].set_title(f'{f} - {b} - Original')
                            axs[row_idx, 0].axis('off')
                            
                            axs[row_idx, 1].imshow(rl_set[row_idx], cmap='gray')
                            axs[row_idx, 1].set_title(f'{b} - CD with {operacao}')
                            axs[row_idx, 1].axis('off')
                            
                            axs[row_idx, 2].imshow(rl_set_obr[row_idx], cmap='gray')
                            axs[row_idx, 2].set_title(f'{b} OR - CD with {operacao}')
                            axs[row_idx, 2].axis('off')
                            
                            axs[row_idx, 3].imshow(masks[row_idx], cmap='gray')
                            axs[row_idx, 3].set_title(f'Mask')
                            axs[row_idx, 3].axis('off')
                            
                            
                        # Ajustar layout e exibir se necessário
                        plt.tight_layout()
                        #plt.show()
            
                        # Salvar a figura no PDF
                        pdf.savefig(fig, dpi=300)
                            
                        plt.close(fig)  # Fechar a figura para liberar memória
                        
                        #reinicia as listas
                        imagens = []
                        masks = []
                        rl_set = []
                        rl_set_obr = []

        
        # Cria figura com 2 subplots lado a lado
        cores = {
            'b2': '#1f77b4',
            'b3': '#2ca02c',
            'b4': '#d62728',
            'b8': '#9467bd',
            'ndvi': '#ff7f0e',
            'ndwi': '#8c564b',
            'ndbi': '#e377c2'}
                    
        #### IOU Boxplot Graphs ####
        fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=False)
        
        # --- IOU normal ---
        data1 = []
        for b in band_list:
            for value in iou_hist[b]:
                data1.append({'Banda': b, 'IoU': value})
        df1 = pd.DataFrame(data1)
        
        sns.boxplot(data=df1, x='Banda', y='IoU', ax=axes[0], palette=cores)
        axes[0].set_title(f'IoU per band - CD over {operacao} (%)')
        axes[0].set_xlabel('Band')
        axes[0].set_ylabel('IoU')
        axes[0].grid(True)
        
        resumo1 = df1.groupby('Banda')['IoU'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        print("\nResumo estatístico da métrica IoU por erosão por banda:\n")
        print(resumo1.round(4))
        
        
        # --- IOU OBR ---
        data2 = []
        for b in band_list:
            for value in obr_iou_hist[b]:
                data2.append({'Banda': b, 'IoU': value})
        df2 = pd.DataFrame(data2)
        
        sns.boxplot(data=df2, x='Banda', y='IoU', ax=axes[1], palette=cores)
        axes[1].set_title(f'IoU per OR band - CD over {operacao} (%)')
        axes[1].set_xlabel('Band')
        axes[1].set_ylabel('')
        axes[1].grid(True)
        
        resumo2 = df2.groupby('Banda')['IoU'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        print(f"\nResumo estatístico da métrica IoU_OBR por {operacao} por banda:\n")
        print(resumo2.round(4))
        
        #plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.show()
        plt.close(fig)
        
        #### F1-score Boxplot Graphs ####
        fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=False)
        # --- F1-Score normal ---
        data3 = []
        for b in band_list:
            for value in f1_hist[b]:
                data3.append({'Banda': b, 'F1': value})
        df3 = pd.DataFrame(data3)
        
        sns.boxplot(data=df3, x='Banda', y='F1', ax=axes[0], palette=cores)
        axes[0].set_title(f'F1-Score per band - CD over {operacao} (%)')
        axes[0].set_xlabel('Band')
        axes[0].set_ylabel('F1-Score')
        axes[0].grid(True)
        
        resumo3 = df3.groupby('Banda')['F1'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        print(f"\nResumo estatístico da métrica F1-Score por {operacao} por banda:\n")
        print(resumo3.round(4))
        
        # --- F1-Score OBR ---
        data4 = []
        for b in band_list:
            for value in obr_f1_hist[b]:
                data4.append({'Banda': b, 'F1': value})
        df4 = pd.DataFrame(data4)
        
        sns.boxplot(data=df4, x='Banda', y='F1', ax=axes[1], palette=cores)
        axes[1].set_title(f'F1-Score per OR band - CD over {operacao} (%)')
        axes[1].set_xlabel('Band')
        axes[1].set_ylabel('')
        axes[1].grid(True)
        
        resumo4 = df4.groupby('Banda')['F1'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        print(f"\nResumo estatístico da métrica F1-Score_OBR por {operacao} por banda:\n")
        print(resumo4.round(4))
        
        #plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.show()
        plt.close(fig)
        
        
        #### IOU Barplot Graphs ####
        # Gráficos de barras com valores sobrepostos para as médias por banda
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        
        # --- Barplot IoU ---
        mean_iou = df1.groupby('Banda')['IoU'].mean().reset_index()
        
        sns.barplot(data=mean_iou, x='Banda', y='IoU', hue='Banda', ax=axes[0], palette=cores, legend=False)

        axes[0].set_title(f'Mean IoU per band - CD over {operacao} (%)')
        axes[0].set_xlabel('Band')
        axes[0].set_ylabel('IoU')
        #axes[0].grid(True)
        
        # Adiciona os valores sobre as barras
        for i, row in mean_iou.iterrows():
            axes[0].text(i, row['IoU'] + 0.01, f"{row['IoU']:.2f}", ha='center', va='bottom')
        
        # --- Barplot IoU OBR ---
        mean_iou_obr = df2.groupby('Banda')['IoU'].mean().reset_index()
        
        sns.barplot(data=mean_iou_obr, x='Banda', y='IoU', hue='Banda', ax=axes[1], palette=cores, legend=False)

        axes[1].set_title(f'Mean IoU per OR band - CD over {operacao} (%)')
        axes[1].set_xlabel('Band')
        axes[1].set_ylabel('')
        #axes[1].grid(True)
        
        for i, row in mean_iou_obr.iterrows():
            axes[1].text(i, row['IoU'] + 0.01, f"{row['IoU']:.2f}", ha='center', va='bottom')
            
        #plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.show()
        plt.close(fig)
            

        #### F1-Score Barplot Graphs ####
        # Gráficos de barras com valores sobrepostos para as médias por banda
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        
        # --- Barplot F1-Score ---
        mean_f1 = df3.groupby('Banda')['F1'].mean().reset_index()
        
        sns.barplot(data=mean_f1, x='Banda', y='F1', hue='Banda', ax=axes[0], palette=cores, legend=False)
        
        axes[0].set_title(f'Mean F1-Score per band - CD over {operacao} (%)')
        axes[0].set_xlabel('Banda')
        axes[0].set_ylabel('F1-Score')
        #axes[0].grid(True)
        
        for i, row in mean_f1.iterrows():
            axes[0].text(i, row['F1'] + 0.01, f"{row['F1']:.2f}", ha='center', va='bottom')
        
        # --- Barplot F1-Score OBR ---
        mean_f1_obr = df4.groupby('Banda')['F1'].mean().reset_index()
        
        sns.barplot(data=mean_f1_obr, x='Banda', y='F1', hue='Banda', ax=axes[1], palette=cores, legend=False)

        axes[1].set_title(f'Mean F1-Score per OR band - CD over {operacao} (%)')
        axes[1].set_xlabel('Banda')
        axes[1].set_ylabel('')
        #axes[1, 1].grid(True)
        
        for i, row in mean_f1_obr.iterrows():
            axes[1].text(i, row['F1'] + 0.01, f"{row['F1']:.2f}", ha='center', va='bottom')
        
        #plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.show()
        plt.close(fig)
        
        # Identifica a melhor banda para cada métrica
        melhor_iou = mean_iou.loc[mean_iou['IoU'].idxmax()]
        melhor_iou_obr = mean_iou_obr.loc[mean_iou_obr['IoU'].idxmax()]
        melhor_f1 = mean_f1.loc[mean_f1['F1'].idxmax()]
        melhor_f1_obr = mean_f1_obr.loc[mean_f1_obr['F1'].idxmax()]
        
        # Cria uma nova figura com o texto
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis('off')
        
        texto = f"""
        Best bands by metric – RL using {operacao}
        
        IoU (normal):
          Band: {melhor_iou['Banda']}
          Mean value: {melhor_iou['IoU']:.4f}
        
        IoU (OBR):
          Band: {melhor_iou_obr['Banda']}
          Mean value: {melhor_iou_obr['IoU']:.4f}
        
        F1-Score (normal):
          Band: {melhor_f1['Banda']}
          Mean value: {melhor_f1['F1']:.4f}
        
        F1-Score (OBR):
          Band: {melhor_f1_obr['Banda']}
          Mean value: {melhor_f1_obr['F1']:.4f}
"""
        
        # Adiciona o texto centralizado
        ax.text(0.5, 0.5, texto, ha='right', va='center', fontsize=12, family='monospace')
        
        # Salva no PDF
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
 

def Morphological_Linear_Features_Enhance_S2_imgs(
    img_dir,
    output_dir,
    band,
    operacao,
    length_list,
    NS,
    ste=np.ones((3, 3), np.int8),
    thickness=1,
    p_processing=True,
    lp=95,
    channel_gmm=None,
    skl_analysis=True,
    gmm_analysis=False,
    gmm_mdl=None,
    scaler=None,
    conditional_dil=False,
    save_results=True,
    show_imgs=False):
    """
    Aplica realce linear em imagens Sentinel-2, com pós-processamento
    baseado em esqueletos, e avalia métricas de segmentação e eixo viário.
    """

    band_dir = os.path.join(img_dir, "Imgs", band)
    image_files = sorted([
        f for f in os.listdir(band_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ])

    metrics_hist = {
        "img_nm":[],
        "length_B":[],
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "skl_iou": [],
        "skl_precision": [],
        "skl_recall": [],
        "skl_f1": [],
        "bf_score": [],
        "median_length_skl":[]
        }
    
    for length in length_list:
        print(f'Elemento estrururante linear B de {length} pixels.')
        
        if save_results:
            out_dir = os.path.join(output_dir, band, str(length))
            os.makedirs(out_dir, exist_ok=True)

        for f in tqdm(image_files, desc=f"Linear enhancement – band {band}"):
    
            # --------------------------------------------------
            # Máscara de referência
            # --------------------------------------------------
            mask_ref = np.array(Image.open(os.path.join(img_dir, "Mask", f)).convert("L")) > 0
    
            # --------------------------------------------------
            # Imagem Sentinel-2
            # --------------------------------------------------
            if channel_gmm:
                img_s = None
                meta = None
                
                for i, ch in enumerate(channel_gmm):
                    img_path = os.path.join(img_dir, "Imgs", ch, f)
                
                    with rasterio.open(img_path) as src:
                        band_img = src.read(1).astype(np.float32)
                
                        # Inicialização do array empilhado
                        if img_s is None:
                            H, W = band_img.shape
                            img_s = np.zeros((len(channel_gmm), H, W), dtype=np.float32)
                
                            # Metadados de referência (usar sempre a primeira banda)
                            meta = src.meta.copy()
                        else:
                            # Verificação de consistência espacial
                            assert band_img.shape == (H, W), f"Incompatibilidade espacial em {ch}/{f}"
                            assert src.transform == meta["transform"], f"Transform diferente em {ch}/{f}"
                            assert src.crs == meta["crs"], f"CRS diferente em {ch}/{f}"
                
                        img_s[i] = band_img  
    
            #abre a imagem a ser utilizada no realce linear        
            img_path = os.path.join(band_dir, f)
            with rasterio.open(img_path) as src:
                img_b = src.read(1)
                meta = src.meta.copy()
            
            #plt.title(f"Img_original {f}")
            # plt.imshow(img_b, cmap="gray")
            # plt.axis("off")
            # plt.show()
            
            # --------------------------------------------------
            # Pré-processamento opcional
            # --------------------------------------------------
            if p_processing:
                img_b = cv2.GaussianBlur(img_b, (5,5), 0)
                
            # plt.title(f)
            # plt.imshow(img_b, cmap="gray")
            # plt.axis("off")
            # plt.show()
    
            # --------------------------------------------------
            # Realce linear
            # --------------------------------------------------
            img_enhanced = linear_enhance(
                img_b, length, operacao,show_imgs=True,
                thickness=thickness
            )
            
            #plt.title(f"Img_Enhanced {f}")
            # plt.imshow(img_enhanced, cmap="gray")
            # plt.axis("off")
            # plt.show()
    
            # --------------------------------------------------
            # Pós-processamento estrutural
            # --------------------------------------------------
            if skl_analysis:
                img_skl, img_skl_stats = skeleton_analysis(
                    img_enhanced, ste=ste, length_percentile=lp,min_seg=5)
                
                #plt.title(f"Img_Enhanced_Skl {f}")
                # plt.imshow(img_skl, cmap="gray")
                # plt.axis("off")
                # plt.show()
            else:
                img_skl = img_enhanced
                img_skl_stats = {}
                img_skl_stats["median_length"] = 0
                
                            
            # ---------------------------------------------------
            # Filtragem de Dados com modelagem probabilística GMM
            # ---------------------------------------------------
            if gmm_analysis:
                #normaliza a imagem
                if img_s.max() > 1:
                    img_s = (img_s - img_s.min()) / (img_s.max() - img_s.min() + 1e-8)
                
                img_gmm = gmm_filter_mask(img_s, img_skl, gmm_mdl)
                
                # plt.title(f"Img_GMM {f}")
                # plt.imshow(img_gmm, cmap="gray")
                # plt.axis("off")
                # plt.show()
                
            else:
                img_gmm = img_skl
                
            
            # ---------------------------------------------------
            # Dilatação Condicional
            # ---------------------------------------------------
            if conditional_dil:
    
                #aplica a dilatação condicional
                rec_img = ConditionalDilation(img_gmm, img_b)
                
                # plt.title(f"Img_DC {f}")
                # plt.imshow(rec_img, cmap="gray")
                # plt.axis("off")
                # plt.show()
                
            else:
                rec_img = img_gmm
    
            #plt.title(f"Img_DC {f}")
            # plt.imshow(rec_img, cmap="gray")
            # plt.axis("off")
            # plt.show()
            
            #########################################################
            img_bin = (rec_img > 0).astype(np.uint8)
    
            # imagem de saída (uint8) APENAS para salvamento
            img_out = (img_bin * 255)
    
            # --------------------------------------------------
            # Salvamento com georreferenciamento
            # --------------------------------------------------
            if save_results:
                meta_out = meta.copy()
                meta_out.update({
                    "dtype": "uint8",
                    "count": 1,
                    "nodata": 0})
    
                for k in [
                    "tiled", "compress", "interleave",
                    "photometric", "blockxsize", "blockysize"
                ]:
                    meta_out.pop(k, None)
    
                with rasterio.open(
                    os.path.join(out_dir, f), "w", **meta_out
                ) as dst:
                    dst.write(img_bin, 1)
    
    
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
                
            
            (skl_iou, skl_prec, skl_rec, skl_f1, _, _, _, _) = Calc_Metrics(img_bin, skl_mask_ref)
    
            metrics_hist["img_nm"].append(f)
            metrics_hist["length_B"].append(length)
            metrics_hist["iou"].append(iou)
            metrics_hist["f1"].append(f1)
            metrics_hist["precision"].append(precision)
            metrics_hist["recall"].append(recall)
            metrics_hist["skl_iou"].append(skl_iou)
            metrics_hist["skl_precision"].append(skl_prec)
            metrics_hist["skl_recall"].append(skl_rec)
            metrics_hist["skl_f1"].append(skl_f1)
            metrics_hist["bf_score"].append(bf_score)
            metrics_hist["median_length_skl"].append(img_skl_stats["median_length"])
    
            # --------------------------------------------------
            # Visualização opcional
            # --------------------------------------------------
            if show_imgs:
                plt.figure(figsize=(15, 8))
    
                plt.subplot(1, 3, 1)
                #plt.title(f"{band} – Original")
                plt.imshow(img_s[2,:,:], cmap="gray")#, vmin=0, vmax=1)
                plt.axis("off")

                plt.subplot(1, 3, 2)
                #plt.title("Linear enhancement")
                plt.imshow(mask_ref, cmap="gray")
                plt.axis("off")
    
                plt.subplot(1, 3, 3)
                #plt.title("Linear enhancement")
                plt.imshow(img_out, cmap="gray")
                plt.axis("off")
    
                plt.tight_layout()
                
                out_dir = os.path.join(output_dir, band, 'outputs')
                
                if not os.path.exists(out_dir):                
                    os.makedirs(out_dir, exist_ok=True)
                
                plt.savefig(os.path.join(out_dir, f"{f}_image.png"), dpi=300)
                plt.show()
            

    return metrics_hist


def metrics_dict_to_dataframe(metrics_dict, operation):
    """
    metrics_dict: {band: metrics_hist}
    operation: str ("erosion" ou "dilation")
    """
    rows = []

    for band, metrics in metrics_dict.items():
        n = len(next(iter(metrics.values())))

        for i in range(n):
            row = {
                "band": band,
                "operation": operation,
                "image_id": i
            }
            for metric_name, values in metrics.items():
                row[metric_name] = values[i]

            rows.append(row)

    return pd.DataFrame(rows)


def plot_metrics(
    metric_list,
    metric_name="IoU",
    label="Band",
    title=None,
    ylabel=None,
    figsize=(6, 6),
    show_stats=True
):
    """
    Gera boxplot para uma métrica representada por uma lista.

    Parameters
    ----------
    metric_list : list
        Lista de valores da métrica (ex.: iou_hist, f1_hist)
    metric_name : str
        Nome da métrica (ex.: 'IoU', 'F1-Score')
    label : str
        Rótulo categórico (ex.: nome da banda)
    title : str
        Título do gráfico
    ylabel : str
        Rótulo do eixo Y
    figsize : tuple
        Tamanho da figura
    show_stats : bool
        Exibe resumo estatístico no console
    """

    # --- DataFrame ---
    df = pd.DataFrame({
        "Category": [label] * len(metric_list),
        metric_name: metric_list
    })

    # --- Plot ---
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="Category",
        y=metric_name,
        width=0.4,
        showfliers=True
    )

    plt.xlabel("")
    plt.ylabel(ylabel if ylabel else metric_name)
    plt.title(title if title else f"{metric_name} distribution – {label}")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- Estatísticas ---
    if show_stats:
        resumo = df[metric_name].agg(
            ['count', 'mean', 'median', 'std', 'min', 'max']
        )
        print(f"\nResumo estatístico – {metric_name} ({label}):\n")
        print(resumo.round(4))
    

def Length_Test(ds_dir, band, operacao, B, length, skl_analysis=True, LP = 95):
    image_files = sorted([
        f for f in os.listdir(os.path.join(ds_dir, "Imgs", band))
        if f.lower().endswith((".tif", ".tiff"))
    ])

    metrics = {
        "iou": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "skl_iou": [],
        "skl_f1": [],
        "skl_prec": [],
        "skl_rec": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "tn": []
    }

    for f in image_files:

        # --- Máscara de referência ---
        mask_ref = np.array(
            Image.open(os.path.join(ds_dir, "Mask", f)).convert("L")
        ) > 0

        # --- Banda ---
        img = tiff.imread(os.path.join(ds_dir, "Imgs", band, f))
        if img.ndim > 2:
            img = img[:, :, 0]

        # --- Realce linear ---
        img_enhanced = linear_enhance(img, length, operacao)

        # --- Pós-processamento ---
        if skeleton_analysis:
            img_pp, _ = skeleton_analysis(img_enhanced, ste=B, dilation=True, seg_filter=True,length_percentile=LP)
        else:
            img_pp = img_enhanced

        # --- Métricas ---
        (iou, precision, recall, f1, tp, fp, fn, tn) = Calc_Metrics(img_pp, mask_ref)

        metrics["iou"].append(iou)
        metrics["f1"].append(f1)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["tp"].append(tp)
        metrics["fp"].append(fp)
        metrics["fn"].append(fn)
        metrics["tn"].append(tn)
        
        # --- Métricas Skl ---
        skl_mask_ref = skeletonize(mask_ref, method="zhang")
            
        (skl_iou, skl_prec, skl_rec, skl_f1, _, _, _, _) = Calc_Metrics(img_pp, skl_mask_ref)
        
        metrics["skl_iou"].append(skl_iou)
        metrics["skl_f1"].append(skl_f1)
        metrics["skl_prec"].append(skl_prec)
        metrics["skl_rec"].append(skl_rec)

    # Mediana por métrica
    return {k: np.nanmedian(v) for k, v in metrics.items()}

def Test_Length_process(img_dir, output_dir, length_list, band_list, operation, ste, skl_a=True,lp=95):

    all_metrics = [
        "iou", "f1", "precision", "recall",
        "skl_iou", "skl_f1", "skl_prec", "skl_rec", "tp", "fp", "fn", "tn"]


    results = {
        metric: {b: [] for b in band_list}
        for metric in all_metrics
    }

    # --------------------------------------------------
    # Experimentos
    # --------------------------------------------------
    for length in length_list:
        print(f"- Comprimento B: {length}")
        for band in band_list:
            print(f"    Banda: {band}")

            metrics = Length_Test(
                img_dir, band, operation, ste, length, skl_analysis=skl_a, LP=lp)

            for m in all_metrics:
                results[m][band].append(metrics[m])

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
        
    # --------------------------------------------------
    # Melhor comprimento por métrica
    # --------------------------------------------------
    for band in band_list:
        print(f"\nBanda: {band}")
        for metric in ["iou", "f1", "skl_iou"]:
            values = results[metric][band]
            best_idx = np.nanargmax(values)
            print(
                f"  Best {metric.upper()}: "
                f"B = {length_list[best_idx]} "
                f"({values[best_idx]:.4f})"
            )

    return results

def Sklthr_Test(ds_dir, band, operacao, B, length,lp):
    image_files = sorted([
        f for f in os.listdir(os.path.join(ds_dir, "Imgs", band))
        if f.lower().endswith((".tif", ".tiff"))
    ])

    metrics = {
        "iou": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "skl_iou": [],
        "skl_f1": [],
        "skl_prec": [],
        "skl_rec": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "tn": []
    }

    for f in image_files:

        # --- Máscara de referência ---
        mask_ref = np.array(
            Image.open(os.path.join(ds_dir, "Mask", f)).convert("L")
        ) > 0

        # --- Banda ---
        img = tiff.imread(os.path.join(ds_dir, "Imgs", band, f))
        if img.ndim > 2:
            img = img[:, :, 0]

        # --- Realce linear ---
        img_enhanced = linear_enhance(img, length, operacao)

        # --- Pós-processamento ---
        img_pp, _ = skeleton_analysis(
            img_enhanced, ste=B, dilation=True, length_percentile=lp
        )

        # --- Métricas ---
        (iou, precision, recall, f1, tp, fp, fn, tn) = Calc_Metrics(img_pp, mask_ref)

        metrics["iou"].append(iou)
        metrics["f1"].append(f1)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["tp"].append(tp)
        metrics["fp"].append(fp)
        metrics["fn"].append(fn)
        metrics["tn"].append(tn)
        
        # --- Métricas Skl ---
        skl_mask_ref = skeletonize(mask_ref, method="zhang")
            
        (skl_iou, skl_prec, skl_rec, skl_f1, _, _, _, _) = Calc_Metrics(img_pp, skl_mask_ref)
        
        metrics["skl_iou"].append(skl_iou)
        metrics["skl_f1"].append(skl_f1)
        metrics["skl_prec"].append(skl_prec)
        metrics["skl_rec"].append(skl_rec)

    # Mediana por métrica
    return {k: np.nanmedian(v) for k, v in metrics.items()}

#Rotina de teste de limiar de corte dos esqueletos
def Test_SklThr_process(img_dir, output_dir, length, thr_list, band_list, operation, ste):

    all_metrics = [
        "iou", "f1", "precision", "recall",
        "skl_iou", "skl_f1", "skl_prec", "skl_rec", "tp", "fp", "fn", "tn"]


    results = {
        metric: {b: [] for b in band_list}
        for metric in all_metrics
    }

    # --------------------------------------------------
    # Experimentos
    # --------------------------------------------------
    for threshold in thr_list:
        print(f"- Limiar: {threshold}")
        for band in band_list:
            print(f"    Banda: {band}")

            metrics = Sklthr_Test(
                img_dir, band, operation, ste, length, threshold
            )

            for m in all_metrics:
                results[m][band].append(metrics[m])

    # --------------------------------------------------
    # Gráficos
    # --------------------------------------------------
    #inglês
    for metric in all_metrics:
        plt.figure(figsize=(10, 6))
        for band in band_list:
            plt.plot(
                thr_list,
                results[metric][band],
                marker="o",
                label=f"Band {band}"
            )

        plt.xlabel("Percentile Threshold")
        plt.ylabel(metric.replace("_", " ").upper())
        plt.title(f"{metric.replace('_', ' ').upper()} vs Threshold")
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
                thr_list,
                results[metric][band],
                marker="o",
                label=f"Banda {band}"
            )

        plt.xlabel("Limiar Percentil")
        plt.ylabel(metric.replace("_", " ").upper())
        plt.title(f"{metric.replace('_', ' ').upper()} X Limiar")
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
        
    # --------------------------------------------------
    # Melhor comprimento por métrica
    # --------------------------------------------------
    for band in band_list:
        print(f"\nBanda: {band}")
        for metric in ["iou", "f1", "skl_iou"]:
            values = results[metric][band]
            best_idx = np.nanargmax(values)
            print(
                f"  Best {metric.upper()}: "
                f"B = {thr_list[best_idx]} "
                f"({values[best_idx]:.4f})"
            )

    return results

def summarize_length_results(results, length_list, band_list):
    metrics = list(results.keys())

    # Tabelas vazias
    df_best_value = pd.DataFrame(index=metrics, columns=band_list, dtype=float)
    df_best_length = pd.DataFrame(index=metrics, columns=band_list, dtype=float)

    for metric in metrics:
        for band in band_list:
            values = np.array(results[metric][band])

            if np.all(np.isnan(values)):
                df_best_value.loc[metric, band] = np.nan
                df_best_length.loc[metric, band] = np.nan
                continue

            best_idx = np.nanargmax(values)

            df_best_value.loc[metric, band] = values[best_idx]
            df_best_length.loc[metric, band] = length_list[best_idx]

    # Nomes mais legíveis (opcional)
    df_best_value.index = [m.upper() for m in df_best_value.index]
    df_best_length.index = [m.upper() for m in df_best_length.index]

    return df_best_value, df_best_length


def restrict_by_distance(markers, max_dist):
    """
    Cria uma máscara espacial limitada ao entorno dos marcadores.
    """
    dist = distance_transform_edt(1 - markers)
    return (dist <= max_dist).astype(np.uint8)
  
  
def conditional_dilation(marker_bin, mask_gray):
    """
    Reconstrução morfológica por dilatação (dilatação condicional).

    Parameters
    ----------
    marker_bin : np.ndarray (uint8 ou bool)
        Marcadores binários (0/1).
    mask_gray : np.ndarray (float ou uint8)
        Imagem em níveis de cinza (máscara da reconstrução).

    Returns
    -------
    rec : np.ndarray
        Imagem reconstruída em níveis de cinza.
    """

    # Garantia de tipo
    marker = marker_bin.astype(np.float32)

    # Ajuste: marcador deve ser <= máscara
    marker = np.minimum(marker, mask_gray)

    # Reconstrução por dilatação
    rec = reconstruction(
        seed=marker,
        mask=mask_gray,
        method='dilation'
    )

    return rec


def ConditionalDilation(
    mask,
    img_1ch,
    m_dist=3,
    pctls=(25, 90),
    spatial_lim=True,
    spectral_lim=True
):
    # --- Marcadores binários ---
    marker_bin = (mask > 0).astype(np.uint8)

    if marker_bin.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # # --- Normalização ---
    # img_norm = img_1ch.astype(np.float32)
    # img_norm -= img_norm.min()
    # if img_norm.max() > 0:
    #     img_norm /= img_norm.max()
    img_norm = img_1ch
    
    # --------------------------------------------------
    # Restrição espacial
    # --------------------------------------------------
    if spatial_lim:
        spatial_mask = restrict_by_distance(
            marker_bin, max_dist=m_dist
        ).astype(bool)
    else:
        spatial_mask = np.ones_like(marker_bin, dtype=bool)

    # --------------------------------------------------
    # Restrição espectral
    # --------------------------------------------------
    if spectral_lim:
        marker_vals = img_norm[marker_bin > 0]

        if marker_vals.size == 0:
            return np.zeros_like(mask, dtype=np.uint8)

        low, high = np.percentile(marker_vals, pctls)
        spectral_mask = (img_norm >= low) & (img_norm <= high)
    else:
        spectral_mask = np.ones_like(marker_bin, dtype=bool)

    # --------------------------------------------------
    # Máscara final (reconstrução em tons de cinza)
    # --------------------------------------------------
    final_mask = img_norm * spatial_mask * spectral_mask

    # --- Dilatação condicional / reconstrução ---
    rec_img = conditional_dilation(marker_bin, final_mask)

    return rec_img

# Função para extrair características de uma imagem
def extract_features(image):
    # Verificar o número de canais da imagem
    if len(image.shape) == 2:  # Imagem de 1 canal
        features = image.flatten()  # Achatar para um vetor 1D
    elif len(image.shape) == 3:  # Imagem de mais de 1 canal (RGB)
        # Combinar os canais em um único vetor de características
        # --- Reorganiza para (512, 512, 6)
        image = np.transpose(image, (1, 2, 0))
        features = image.reshape (-1, image.shape[2])

    return features


def gmm_filter_mask(
    image,
    mask_bin,
    model
):

    H, W = mask_bin.shape
    mask_flat = mask_bin.ravel().astype(bool)

    # --- Extração de características ---
    features = extract_features(image)
    features_sel = features[mask_flat]

    if features_sel.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # --- Log-verossimilhança ---
    log_p = model.score_samples(features_sel)

    # --- Aplicação do limiar aprendido no treino ---
    keep = (log_p >= model.prob_thr_).astype(np.uint8)

    # --- Reconstrução da máscara ---
    filtered_mask = np.zeros(H * W, dtype=np.uint8)
    filtered_mask[mask_flat] = keep

    return filtered_mask.reshape(H, W)