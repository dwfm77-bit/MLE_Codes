#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:01:27 2024

@author: DWander
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
#from skimage.morphology import opening, rectangle, square, disk, skeletonize
from skimage.draw import line as draw_line
from skimage.transform import rotate
from skimage.util import view_as_windows
from skimage.measure import regionprops
import scipy.ndimage as ndi
import skimage.measure as skm
from skimage.morphology import skeletonize, reconstruction
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation, distance_transform_edt
#from pathlib import Path
from PIL import Image#, ImageDraw
import numpy as np
from utils.morph import mm
import cv2
import time
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import os, random
import rasterio
#import copy
#import math
import tifffile as tiff
from tqdm import tqdm
#from datetime import datetime
import shutil
import seaborn as sns
from sklearn.model_selection import KFold
from itertools import combinations
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, jaccard_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight

############################################################################

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

def confusion_elements(pred, ref):
    """
    pred, ref: arrays booleanos (True = estrada)
    """
    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    fn = np.logical_and(~pred, ref).sum()
    tn = np.logical_and(~pred, ~ref).sum()
    return tp, fp, fn, tn

def Calc_Metrics(pred, mask_ref):
    
    # --- Classical Metrics Predictions ---
    tp, fp, fn, tn = confusion_elements(pred, mask_ref)
    
    iou = iou_np(pred, mask_ref)
    precision = precision_np(pred, mask_ref)
    recall = recall_np(pred, mask_ref)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
    return iou, precision, recall, f1_score, tp, fp, fn, tn

# Boxplots/Distribuição das métricas
def MetricsBoxplot(acc_list, f1_list, iou_list, recall_list, precision_list, out_folder, Box_Labels = ["Acc", "F1", "IoU", "Recall", "Precision"]):

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # paleta personalizada
    
    box = plt.boxplot(
        [acc_list, f1_list, iou_list, recall_list, precision_list],
        labels=Box_Labels,
        patch_artist=True  # permite colorir o interior das caixas
    )
    
    # Aplicar cores nas caixas
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)  # transparência opcional
    
    # Estilo das medianas
    for median in box['medians']:
        median.set(color="black", linewidth=2)
    
    plt.title("Distribuição das métricas nos testes")
    plt.savefig(os.path.join(out_folder, "metrics_distribution_colored.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# Matriz de confusão do melhor modelo
def ConfusionMatrix(conf_matrix, out_folder):
    plt.figure(figsize=(4,4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Não-Estrada", "Estrada"],
                yticklabels=["Não-Estrada", "Estrada"])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão (Melhor Modelo)")
    plt.savefig(os.path.join(out_folder, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
# Curva Precisão-Recall
def PrecisionRecallCurve(best_labels, best_probs, out_folder):
    prec, rec, _ = precision_recall_curve(best_labels, best_probs)
    ap = average_precision_score(best_labels, best_probs)
    
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precisão-Recall (Melhor Modelo)")
    plt.legend()
    plt.savefig(os.path.join(out_folder, "precision_recall_curve.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

#Curva ROC
def RocCurve(best_labels, best_probs, out_folder):
    fpr, tpr, _ = roc_curve(best_labels, best_probs)
    auc = roc_auc_score(best_labels, best_probs)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Curva ROC (Melhor Modelo)")
    plt.legend()
    plt.savefig(os.path.join(out_folder, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def extract_features_old(image):
    # Verificar o número de canais da imagem
    if len(image.shape) == 2:  # Imagem de 1 canal
        features = image.flatten()  # Achatar para um vetor 1D
    elif len(image.shape) == 3:  # Imagem de mais de 1 canal (RGB)
        # Combinar os canais em um único vetor de características
        # --- Reorganiza para (512, 512, 6)
        image = np.transpose(image, (1, 2, 0))
        features = image.reshape (-1, image.shape[2])

    return features

def extract_features(image):

    if image.ndim == 2:
        # (H,W) -> (H*W,1)
        features = image.reshape(-1,1)

    elif image.ndim == 3:
        # (C,H,W) -> (H,W,C)
        image = np.transpose(image, (1,2,0))

        # (H,W,C) -> (H*W,C)
        features = image.reshape(-1, image.shape[2])

    else:
        raise ValueError("Formato de imagem inválido")

    return features

def prepare_structural_data(mle_imgs_url, Imgs_List, WS, NS = [512,512], mode="binary"):
    """WS = window size"""

    # Listas separadas para espectral e estrutural
    X_dir_list  = []


    for img in Imgs_List:

        # --- Imagem MLE ---
        img_path = os.path.join(mle_imgs_url, img)
        b_img = tiff.imread(img_path).astype(np.uint16)

        assert b_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {img}: {b_img.shape}"

        # ==================================================
        #features_dir = extract_features_directions(b_img, window_size=WS)
        features_dir = extract_features_directions_vs2(b_img, window_size=WS, mode="binary")  # (H*W, 12)

        # Armazena separadamente
        X_dir_list.append(features_dir)#/255)

    # ==================================================
    # Empilhamento global
    # ==================================================
    X_dir  = np.vstack(X_dir_list)

    return X_dir

#versão inicial
def extract_features_directions(image, window_size=[3,3]):
    '''Tipo

    Descritor direcional completo (alta dimensionalidade).

    Ideia

    Extrai todos os valores dos pixels ao longo de 4 direções dentro da janela.

    Direções consideradas:

        0° — horizontal

    45° — diagonal principal

    90° — vertical

    135° — diagonal secundária

    Estrutura do vetor

    Cada direção mantém todos os pixels da linha direcional.'''

    if window_size[0] % 2 == 0:
        raise ValueError("window_size deve ser ímpar")

    # --- Ajusta formato para (H, W, C)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    elif image.shape[0] < image.shape[-1]:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))

    H, W, C = image.shape
    pad = window_size[0] // 2

    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    # (H, W, wy, wx, C)
    patches = view_as_windows(padded, (window_size[0], window_size[1], C))
    patches = patches.reshape(H, W, window_size[0], window_size[1], C)

    center = pad

    # ---- Seleção direcional ----
    horizontal = patches[:, :, center, :, :]                # 0°
    vertical   = patches[:, :, :, center, :]                # 90°
    diag_45    = patches[:, :, range(window_size[0]), range(window_size[1]), :]  # 45°
    diag_135   = patches[:, :, range(window_size[0]-1, -1, -1), range(window_size[1]), :]  # 135°

    # Reorganiza cada direção para (H, W, window_size*C)
    horizontal = horizontal.reshape(H, W, -1)
    vertical   = vertical.reshape(H, W, -1)
    diag_45    = diag_45.reshape(H, W, -1)
    diag_135   = diag_135.reshape(H, W, -1)

    # Concatena na ordem desejada
    features = np.concatenate(
        [horizontal, diag_45, vertical, diag_135],
        axis=-1
    )

    # Flatten final → (N_pixels, N_features)
    features = features.reshape(-1, features.shape[-1])

    return features

def extract_features_directions_vs0(image,
                                window_size=(3, 3),
                                mode="binary"):
    """
    Gera descritor estrutural compacto direcional.
    
    Ideia

    Não guarda os valores dos pixels.
    Ele mede se existe continuidade estrutural na direção.

    Parameters
    ----------
    image : ndarray (H,W) ou (H,W,C) ou (C,H,W)
    window_size : tuple ímpar (wy, wx)
    mode : "binary" ou "prob"

    Returns
    -------
    features : ndarray (N_pixels, 4)
               Ordem: [0°, 45°, 90°, 135°]
               
    4 features
    4 features
    
    Ordem:
    
    [0°,45°,90°,135°]
    [0°,45°,90°,135°]
                   
    """

    if window_size[0] % 2 == 0:
        raise ValueError("window_size deve ser ímpar")

    if mode not in ["binary", "prob"]:
        raise ValueError("mode deve ser 'binary' ou 'prob'")

    # --- Ajuste formato para (H, W, C)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    elif image.shape[0] < image.shape[-1]:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))

    H, W, C = image.shape
    pad = window_size[0] // 2

    padded = np.pad(image,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode='reflect')

    patches = view_as_windows(
        padded,
        (window_size[0], window_size[1], C)
    )

    patches = patches.reshape(H, W,
                              window_size[0],
                              window_size[1],
                              C)

    center = pad
    total_pixels = window_size[0] * C

    # -------- Seleção Direcional --------
    horizontal = patches[:, :, center, :, :]        # 0°
    vertical   = patches[:, :, :, center, :]        # 90°
    diag_45    = patches[:, :, range(window_size[0]),
                         range(window_size[1]), :]
    diag_135   = patches[:, :, range(window_size[0]-1, -1, -1),
                         range(window_size[1]), :]

    # -------- Contagem de pixels != 0 --------
    def compute_direction_stat(direction_tensor):

        # direction_tensor → (H, W, window_size, C)
        nonzero = (direction_tensor != 0).sum(axis=(2, 3))

        if mode == "binary":
            # 1 se todos != 0
            return (nonzero == total_pixels).astype(np.float32)

        else:  # mode == "prob"
            return (nonzero / total_pixels).astype(np.float32)

    d0   = compute_direction_stat(horizontal)
    d45  = compute_direction_stat(diag_45)
    d90  = compute_direction_stat(vertical)
    d135 = compute_direction_stat(diag_135)

    # Stack final → (H, W, 4)
    features = np.stack([d0, d45, d90, d135], axis=-1)

    # Flatten → (N_pixels, 4)
    features = features.reshape(-1, 4)

    return features



def extract_features_directions_vs1(image,
                                window_size=(3, 3),
                                mode="binary"):
    """
    Descritor estrutural direcional compacto com
    verificação do pixel central.
    
    Tipo

    Descritor estrutural condicionado ao pixel central.
    
    Ideia
    
    Primeiro verifica se o pixel central é ativo.
    
    center=(pixel≠0)
    center=(pixel=0)
    
    Somente então calcula as direções.
    
    Vetor final
    5 features
    5 features
    
    Ordem:
    
    [center,0°,45°,90°,135°]

    Retorna: (N_pixels, 5)
    Ordem: [center, 0°, 45°, 90°, 135°]
    """

    if window_size[0] % 2 == 0:
        raise ValueError("window_size deve ser ímpar")

    if mode not in ["binary", "prob"]:
        raise ValueError("mode deve ser 'binary' ou 'prob'")

    # --- Ajuste formato (H, W, C)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    elif image.shape[0] < image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))

    H, W, C = image.shape
    pad = window_size[0] // 2

    padded = np.pad(image,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode='reflect')

    patches = view_as_windows(
        padded,
        (window_size[0], window_size[1], C)
    )

    patches = patches.reshape(H, W,
                              window_size[0],
                              window_size[1],
                              C)

    center = pad

    # -------------------------------------------------
    # Pixel central
    # -------------------------------------------------
    center_pixel = patches[:, :, center, center, :]
    center_mask = (center_pixel != 0).any(axis=2).astype(np.float32)
    # shape → (H, W)

    # Inicializa direções com zeros
    d0 = np.zeros((H, W), dtype=np.float32)
    d45 = np.zeros((H, W), dtype=np.float32)
    d90 = np.zeros((H, W), dtype=np.float32)
    d135 = np.zeros((H, W), dtype=np.float32)

    # -------------------------------------------------
    # Processa apenas onde pixel central é ativo
    # -------------------------------------------------
    active_idx = center_mask == 1

    if np.any(active_idx):

        total_pixels = window_size[0] * C

        horizontal = patches[:, :, center, :, :]
        vertical   = patches[:, :, :, center, :]
        diag_45    = patches[:, :, range(window_size[0]),
                             range(window_size[1]), :]
        diag_135   = patches[:, :, range(window_size[0]-1, -1, -1),
                             range(window_size[1]), :]

        def compute_direction(direction_tensor):

            # reshape robusto
            direction_tensor = direction_tensor.reshape(H, W, -1)

            nonzero = (direction_tensor != 0).sum(axis=2)

            if mode == "binary":
                values = (nonzero == direction_tensor.shape[2]).astype(np.float32)
            else:
                values = (nonzero / direction_tensor.shape[2]).astype(np.float32)

            return values

        d0_all   = compute_direction(horizontal)
        d45_all  = compute_direction(diag_45)
        d90_all  = compute_direction(vertical)
        d135_all = compute_direction(diag_135)

        # Aplica apenas onde centro é ativo
        d0[active_idx]   = d0_all[active_idx]
        d45[active_idx]  = d45_all[active_idx]
        d90[active_idx]  = d90_all[active_idx]
        d135[active_idx] = d135_all[active_idx]

    # -------------------------------------------------
    # Stack final
    # -------------------------------------------------
    features = np.stack(
        [center_mask, d0, d45, d90, d135],
        axis=-1
    )

    features = features.reshape(-1, 5)

    return features


def extract_features_directions_vs2(image,
                                window_size=(3, 3),
                                mode="binary"):
    """
    Descritor estrutural direcional compacto otimizado.
    NÃO usa view_as_windows.

    Retorna: (N_pixels, 5)
    Ordem: [center, 0°, 45°, 90°, 135°]
    """

    if window_size[0] % 2 == 0:
        raise ValueError("window_size deve ser ímpar")

    if mode not in ["binary", "prob"]:
        raise ValueError("mode deve ser 'binary' ou 'prob'")

    # --- Ajuste formato (H, W, C)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    elif image.shape[0] < image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))

    H, W, C = image.shape
    pad = window_size[0] // 2
    L = window_size[0]

    # Máscara binária
    mask = (image != 0)

    # Padding
    padded = np.pad(mask,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode='reflect')

    # Região central
    center = padded[pad:pad+H, pad:pad+W, :]
    center_mask = center.any(axis=2).astype(np.float32)

    # Inicializa acumuladores
    d0_count   = np.zeros((H, W), dtype=np.float32)
    d90_count  = np.zeros((H, W), dtype=np.float32)
    d45_count  = np.zeros((H, W), dtype=np.float32)
    d135_count = np.zeros((H, W), dtype=np.float32)

    # -------------------------------------------------
    # Acumulação direcional via deslocamentos
    # -------------------------------------------------
    for k in range(-pad, pad + 1):

        # Horizontal (0°)
        d0_count += padded[pad:pad+H,
                           pad+k:pad+k+W, :].all(axis=2)

        # Vertical (90°)
        d90_count += padded[pad+k:pad+k+H,
                            pad:pad+W, :].all(axis=2)

        # Diagonal 45°
        d45_count += padded[pad+k:pad+k+H,
                            pad+k:pad+k+W, :].all(axis=2)

        # Diagonal 135°
        d135_count += padded[pad-k:pad-k+H,
                             pad+k:pad+k+W, :].all(axis=2)

    # -------------------------------------------------
    # Normalização / binarização
    # -------------------------------------------------
    if mode == "binary":
        d0   = (d0_count   == L).astype(np.float32)
        d90  = (d90_count  == L).astype(np.float32)
        d45  = (d45_count  == L).astype(np.float32)
        d135 = (d135_count == L).astype(np.float32)
    else:
        d0   = d0_count   / L
        d90  = d90_count  / L
        d45  = d45_count  / L
        d135 = d135_count / L

    # Aplica somente onde centro é ativo
    d0   *= center_mask
    d90  *= center_mask
    d45  *= center_mask
    d135 *= center_mask

    # Stack final
    features = np.stack(
        [center_mask, d0, d45, d90, d135],
        axis=-1
    )

    return features.reshape(-1, 5)


def merge_channels(image1, image2, image3, image4 = None, NS = None):
    """ NS = [height, width] - new size for the images to be merged"""
    
    if NS:
        n_image1 = cv2.resize(image1, (NS[0],NS[1]))
        n_image2 = cv2.resize(image2, (NS[0],NS[1]))
        n_image3 = cv2.resize(image3, (NS[0],NS[1]))
        if image4:
            n_image4 = cv2.resize(image4, (NS[0],NS[1]))
    else:
        # Verificar se todas as imagens têm a mesma dimensão
        assert image1.shape == image2.shape == image3.shape
        if image4:
            n_image4 = cv2.resize(image4, (NS[0],NS[1]))
    
    if image4:
        # Criar uma nova imagem com 4 canais
        merged_image = np.stack((image1, image2, image3, image4), axis=-1)
    else:
        # Criar uma nova imagem com 3 canais
        merged_image = np.stack((image1, image2, image3), axis=-1)
    
    return merged_image

# Move images and masks to the respective folders
def copy_files(files, source_folder, destination_folder):
    for file in files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.copy(source_path, destination_path)
        
def resize_and_save_image(img_path, img_name, output_path, NS, save=True, is_mask=False):
    img = tiff.imread(img_path)
    orig_dtype = img.dtype

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized_img = cv2.resize(img, (NS[1], NS[0]), interpolation=interp)
    resized_img = resized_img.astype(orig_dtype)

    if save:
        os.makedirs(output_path, exist_ok=True)
        tiff.imwrite(
            os.path.join(output_path, img_name),
            resized_img,
            dtype=orig_dtype,
            photometric='minisblack',
            planarconfig='contig'
        )
    else:
        return resized_img

#Recorta as imagens segundo um formato pré-definido
def crop_image(img, new_size):
    lin,col = img.size
    
    # Calcular as coordenadas para o corte central
    left = (col - new_size) // 2
    top = (col - new_size) // 2   
    right = left + new_size       
    bottom = top + new_size       
    
    # Recortar a imagem
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img


def crop_and_save_image(img_path, img_name, output_path, NS, save=True, is_mask=False):
    """
    Recorta a imagem mantendo tipo e qualidade originais.
    - img_path: caminho completo da imagem de entrada
    - img_name: nome do arquivo de saída
    - output_path: diretório de saída
    - NS: (linhas, colunas) do recorte desejado
    - is_mask: usa interpolação nearest se True
    """

    # lê a imagem
    img = tiff.imread(img_path)
    orig_dtype = img.dtype

    # remove dimensões desnecessárias
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    lin, col = img.shape[:2]

    # recorte ou redimensionamento central
    if (lin, col) != tuple(NS):
        if lin >= NS[0] and col >= NS[1]:
            # recorte central
            left = (col - NS[1]) // 2
            top = (lin - NS[0]) // 2
            right = left + NS[1]
            bottom = top + NS[0]
            resized_img = img[top:bottom, left:right]
        else:
            # redimensiona (mantendo tipo)
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
            resized_img = cv2.resize(img, (NS[1], NS[0]), interpolation=interp)
    else:
        resized_img = img

    resized_img = resized_img.astype(orig_dtype)

    if save:
        os.makedirs(output_path, exist_ok=True)
        tiff.imwrite(
            os.path.join(output_path, img_name),
            resized_img,
            dtype=orig_dtype,
            photometric='minisblack',
            planarconfig='contig'
        )
    else:
        return resized_img


#Convert an image to binary
def preprocess_mask(mask):
    
    # Certificar-se de que os valores são 0 ou 1
    n_mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)
    
    # Expandir a dimensão para (height, width, 1)
    #n_mask = np.expand_dims(n_mask, axis=-1)   # [H, W, 1]
    n_mask = np.expand_dims(n_mask, axis=0)   # [1, H, W]
    return n_mask

# Normalização Min-Max
def MinMaxNorm(img):
    img = img.astype('float32')
    return (img - img.min()) / (img.max() - img.min())


# Função para aplicar a Transformada de Hough
def hough_transform(image, threshold=100):
    """
    Aplica a transformada de Hough para detectar linhas retas na imagem.
    
    :param image: Imagem de entrada (em escala de cinza)
    :param threshold: Limite de acúmulo para detectar uma linha
    :return: Imagem com linhas detectadas
    """
    # Verificar se a imagem está em uint8
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # Detecta as linhas usando a Transformada de Hough
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=50, maxLineGap=10)
    
    # Cria uma cópia da imagem para desenhar as linhas
    line_image = np.copy(image)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return line_image

# Função para aplicar o filtro de Laplace
def laplace_filter(image):
    """
    Aplica o filtro de Laplace para detecção de bordas.
    
    :param image: Imagem de entrada (em escala de cinza)
    :return: Imagem com bordas detectadas
    """
    # Verificar se a imagem está em uint8
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # Aplica o filtro de Laplace
    laplace = cv2.Laplacian(image, cv2.CV_64F)
    
    # Converte para uint8 para visualização
    laplace = np.uint8(np.absolute(laplace))
    
    return laplace

#Função que extrai os esqueletos das estradas
def skeletonize_mask(mask, im_show=True):
    # Normaliza para booleano
    mask_bool = mask > 0
    skel = skeletonize(mask_bool)

    if im_show:
        # Plotagem
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
   
        axes[0].imshow(mask, cmap='gray')
        axes[0].set_title("Mascara Original"); axes[0].axis("off")
    
        axes[1].imshow(skel, cmap='gray')
        axes[1].set_title("Mascara Esqueletizada"); axes[1].axis("off")
    
        #fig.savefig(os.path.join(output_folder, f"Teste_{img}.png"))
        plt.show()
        plt.close(fig)

    # Converte para uint8
    return (skel.astype(np.uint8))

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
        #opened = mm.open(image, b = line_se)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, line_se)
        
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

def filter_linear_structures(image, length=3, angle_step= 45, show_results = False):

    orientations = np.arange(0, 180, angle_step)
    responses = []

    for angle in orientations:
        line_se = create_line_se(length, angle)
        line_se = line_se.astype('uint8')
        #print(line_se)
        #opened = mm.open(image, b = line_se)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, line_se)
        
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

def create_line_se(length, angle):
    """Cria um elemento estruturante do tipo linha (agulha) rotacionado."""
    size = length
    se = np.zeros((size, size), dtype=bool)
    rr, cc = draw_line(size // 2, 0, size // 2, size - 1)
    se[rr, cc] = 1
    rotated = rotate(se.astype(float), angle, order=0, preserve_range=True).astype(bool)
    return rotated

def linear_enhance(img, length, operation):
    '''applies conditional dilation operations with linear structuring element in 4 directions (0°, 45°, 90°, 135°)'''

    #LW filter
    h_line = create_line_se(length, 0)
    h_line = h_line.astype(np.uint8)
    h_enhanced = img_compare(img, operation, B = h_line)
        
    #NE filter
    ne_line = create_line_se(length, 45)
    ne_line = ne_line.astype(np.uint8)
    ne_enhanced = img_compare(img, operation, B = ne_line)
    
    #NS filter
    v_line = create_line_se(length, 90)
    v_line = v_line.astype(np.uint8)
    v_enhanced = img_compare(img, operation, B = v_line)
    
    #NO filter
    no_line = create_line_se(length, 135)
    no_line = no_line.astype(np.uint8)
    no_enhanced = img_compare(img, operation, B = no_line)
    
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 4, 1)
    # plt.title("Linear_0°")
    # plt.imshow(h_enhanced, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(1, 4, 2)
    # plt.title("Linear_45°")
    # plt.imshow(ne_enhanced, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(1, 4, 3)
    # plt.title("Linear_90°")
    # plt.imshow(v_enhanced, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(1, 4, 4)
    # plt.title("Linear_135°")
    # plt.imshow(no_enhanced, cmap="gray")
    # plt.axis("off")
    
    # plt.show()

    #Image composition
    img_c = (h_enhanced | v_enhanced | ne_enhanced | no_enhanced).astype(np.uint8)
    
    # plt.imshow(img_c)
    # plt.title("Composed")
    # plt.axis("off")
    # plt.show()
    
    return img_c

def skeleton_analysis(pred_s, ste=None, close=False, length_percentile=80):
    """
    Análise estrutural baseada em esqueletos usando percentis estáveis:
    - remove esqueletos curtos com base em percentil fixo
    - preserva apenas objetos conectados a esqueletos relevantes

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

    if close:
        pred_bin = mm.close(pred_bin, b=ste)
        #pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, ste)

    # --------------------------------------------------
    # 2. Esqueletização
    # --------------------------------------------------
    pred_skl = skeletonize(pred_bin, method="zhang").astype(np.uint8)
    
    # --------------------------------------------------
    # 3. Rotulagem dos esqueletos
    # --------------------------------------------------
    skl_labeled = skm.label(pred_skl, connectivity=2)
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
    lengths = np.array([prop.area for prop in skl_props])
    min_length = 10
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
    pred_labeled = skm.label(pred_bin, connectivity=2)
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
    
    # for stats in pred_skl_stats:
    #     print(f"{stats}: {pred_skl_stats[stats]}")

    return img_post_processed, pred_skl_stats

def LE_Process( img_name, img_dir, out_folder, linear_b, operador, show_results=False, p_processing=True, cls = True, skl_filter=True):

    in_path = os.path.join(img_dir, img_name)
    out_path = os.path.join(out_folder, img_name)

    # --------------------------------------------------
    # Leitura preservando metadados
    # --------------------------------------------------
    with rasterio.open(in_path) as src:
        img_s = src.read(1)              # array 2D
        meta = src.meta.copy()           # metadados completos
        
    # --------------------------------------------------
    # Pré-processamento (opcional)
    # --------------------------------------------------
    if p_processing:
        img_s = cv2.GaussianBlur(img_s, (5,5), 0)

    # --------------------------------------------------
    # Processamento (apenas numpy)
    # --------------------------------------------------
    img_enhanced = linear_enhance(img_s, linear_b, operador)
    img_enhanced = img_enhanced.astype(np.uint8)
    
    #Apply Skeleton filter
    if skl_filter:
        img_enhanced, img_skl_stats = skeleton_analysis(img_enhanced, ste = np.ones((3, 3), dtype='uint8'), close=cls)
    
    # --------------------------------------------------
    # Ajustes do metadata para saída binária
    # --------------------------------------------------
    meta.update({
        "dtype": "uint8",
        "count": 1,
        "nodata": 0
    })

    # Limpeza de campos problemáticos (boa prática)
    for key in [
        "tiled", "compress", "interleave",
        "photometric", "blockxsize", "blockysize"
    ]:
        meta.pop(key, None)

    # --------------------------------------------------
    # Escrita mantendo CRS + transform
    # --------------------------------------------------
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(img_enhanced, 1)

    # --------------------------------------------------
    # Visualização opcional
    # --------------------------------------------------
    if show_results:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title(f"Original – {img_name}")
        plt.imshow(img_s, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Realce Linear – {img_name}")
        plt.imshow(img_enhanced, cmap="gray")
        plt.axis("off")

        plt.show()

    return img_enhanced


def salvar_resultados(image, mask, y_pred_np, prob_classes, H, W, img, 
                          temp_folder, bins, nr_classes):
    """
    Salva visualizações em uma pasta por imagem, sem matplotlib.
    Usa cv2.imwrite para manter a resolução original.
    """

    # Criar pasta específica para a imagem
    img_folder = os.path.join(temp_folder, f"{img}")
    os.makedirs(img_folder, exist_ok=True)

    # 1. Imagem original
    if image.ndim == 3 and image.shape[2] > 1:
        img_to_save = image[:, :, 0]  # salva só a primeira banda
    else:
        img_to_save = image
    cv2.imwrite(os.path.join(img_folder, "Original.png"), img_to_save)

    # 2. Máscara
    cv2.imwrite(os.path.join(img_folder, "Mascara.png"), mask.astype(np.uint8) * 255)

    # 3. Predição binária
    pred_bin = y_pred_np.reshape(H, W).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(img_folder, "Predicao_Binaria.png"), pred_bin)

    # 4. Mapa de probabilidade geral (normalizado para 0–255)
    prob_norm = cv2.normalize(prob_classes.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    prob_norm = prob_norm.astype(np.uint8)
    cv2.imwrite(os.path.join(img_folder, "Mapa_Probabilidade.png"), prob_norm)

    # 5. Intervalos de probabilidade
    for c in range(nr_classes):
        mask_class = (prob_classes == c).astype(np.uint8) * 255
        nome_arquivo = f"Prob_{int(bins[c]*10):02d}-{int(bins[c+1]*10):02d}.png"
        cv2.imwrite(os.path.join(img_folder, nome_arquivo), mask_class)
        
        
def prepare_dataset_vs0(imgs_url_dict, Imgs_List, Labels_List, train_idx, n_total, NS, ch_list, fold_id, t_size=0.2, global_seed=42):
    '''Função de preparação do dataset para o caso de hold out'''

    Imgs_train, Imgs_test, y_train, y_test = train_test_split(Imgs_List, Labels_List, test_size=t_size, random_state=global_seed)
    
    # print(" ")
    # print(f"Conjunto de treinamento: {len(Imgs_train)} imagens")
    # print(f"Conjunto de teste: {len(Imgs_test)} imagens")
    
    #Listas de armazenamento
    Img_train = []
    Label_train = []
    # Img_test = []
    # Label_test = []
    # total_features = 0
    
    #calcula a qtd de imagens para compor o total de amostras desejado
    nr_imgs = int(n_total/(NS[0]*NS[1])) + 1
    
    ## Seleciona um cjt aleatorio de imagens para compor o dataset
    #indices = random.sample(range(len(train_idx)), nr_imgs)
    
    #Gerador aleatório reprodutível por fold
    rng = random.Random(global_seed + fold_id)
    
    #Seleciona índices reais (globais) dentro do fold
    if nr_imgs > len(train_idx):
        nr_imgs = len(train_idx)

    selected_idx = rng.sample(train_idx.tolist(), nr_imgs)

    #Seleciona as imagens e máscaras
    Imgs_Selected = [Imgs_List[i] for i in selected_idx]
    #Labels_Selected = [Labels_List[i] for i in selected_idx]
    
    print(f'- Selecionadas {len(Imgs_Selected)} imagens.')
    
    #for img in tqdm(Imgs_train, desc="Preparando o dataset")
    for idx, img in enumerate(Imgs_Selected):
        #update the mask paths
        label_path = os.path.join(imgs_url_dict['Mask'], img)
        
        #mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(Image.open(label_path))
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        #transforma as mascaras em matriz binaria
        mask_bin = np.where(mask > 0, 1, 0)
        #mask_bin = skeletonize(mask_bin)
        mask_flatten = mask_bin.flatten()

        # plt.imshow(image, cmap = 'gray')
        # plt.axis("off")  # Oculta os eixos
        # plt.show()
    
        # Cria as imagens empilhadas
        image = np.zeros((len(ch_list), NS[0], NS[1]), dtype=np.float32)
        
        for i, ch in enumerate(ch_list):
            img_path = os.path.join(imgs_url_dict[ch], img)
            b_img = tiff.imread(img_path).astype(np.float32)
            
            # if ch == 'mm_ri':
            #     print('mm_ri: ', np.unique(b_img))
            
            # Verifica tamanho
            assert b_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {ch}/{img}: {b_img.shape}"
            
            image[i, :, :] = b_img
        
        features = extract_features(image)
        
        Img_train.append(features)
        Label_train.append(mask_flatten)

    # Converter para arrays
    if len(ch_list) > 1:
        Img_train = np.vstack(Img_train) #empilhamento vertical, aumenta o numero de linhas (* em uma np array cada linha é um cjt de dados entre um [])
        Label_train = np.hstack(Label_train) #empilhamento horizontal, aumenta o numero de colunas (* em uma np array cada coluna é um numero dentro de um [])
    else:
        Img_train = np.array(Img_train)
        Label_train = np.array(Label_train)
        Img_train = Img_train.reshape(-1,1)
        Label_train = Label_train.reshape(-1,1)
        Label_train = Label_train.astype(np.uint8)
        
    # Escalar as características
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(Img_train)
        
    return X_train, Label_train, Imgs_test, y_test, scaler, Imgs_Selected

def prepare_dataset(model, imgs_url_dict, Imgs_List, Labels_List, train_idx, n_total, NS, ch_list, fold_id, no_scale_channels, t_size=0.2, global_seed=42):
    '''Função de preparação do dataset para o caso de bandas já normalizadas'''
    
    Imgs_train, Imgs_test, y_train, y_test = train_test_split(Imgs_List, Labels_List, test_size=t_size, random_state=global_seed)
    
    # print(" ")
    # print(f"Conjunto de treinamento: {len(Imgs_train)} imagens")
    # print(f"Conjunto de teste: {len(Imgs_test)} imagens")
    
    #Listas de armazenamento
    Img_train = []
    Label_train = []
    # Img_test = []
    # Label_test = []
    # total_features = 0
    
    #calcula a qtd de imagens para compor o total de amostras desejado
    nr_imgs = int(n_total/(NS[0]*NS[1])) + 1
    
    ## Seleciona um cjt aleatorio de imagens para compor o dataset
    #indices = random.sample(range(len(train_idx)), nr_imgs)
    
    #Gerador aleatório reprodutível por fold
    rng = random.Random(global_seed + fold_id)
    
    #Seleciona índices reais (globais) dentro do fold
    if nr_imgs > len(train_idx):
        nr_imgs = len(train_idx)

    selected_idx = rng.sample(train_idx.tolist(), nr_imgs)

    #Seleciona as imagens e máscaras
    Imgs_Selected = [Imgs_List[i] for i in selected_idx]
    #Labels_Selected = [Labels_List[i] for i in selected_idx]
    
    print(f'- Selecionadas {len(Imgs_Selected)} imagens.')
    
    scale_cols = []
    no_scale_cols = []
    
    for idx, ch in enumerate(ch_list):
        if ch.lower() in no_scale_channels:
            no_scale_cols.append(idx)
        else:
            scale_cols.append(idx)
            
    
    #for img in tqdm(Imgs_train, desc="Preparando o dataset")
    for idx, img in enumerate(Imgs_Selected):
        #update the mask paths
        label_path = os.path.join(imgs_url_dict['Mask'], img)
        
        #mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(Image.open(label_path))
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        #transforma as mascaras em matriz binaria
        mask_bin = np.where(mask > 0, 1, 0)
        #mask_bin = skeletonize(mask_bin)
        mask_flatten = mask_bin.flatten()

        # plt.imshow(image, cmap = 'gray')
        # plt.axis("off")  # Oculta os eixos
        # plt.show()
    
        # Cria as imagens empilhadas
        image = np.zeros((len(ch_list), NS[0], NS[1]), dtype=np.float32)
        
        for i, ch in enumerate(ch_list):
            img_path = os.path.join(imgs_url_dict[ch], img)
            b_img = tiff.imread(img_path).astype(np.float32)
            
            # if ch == 'mm_ri':
            #     print('mm_ri: ', np.unique(b_img))
            
            # Verifica tamanho
            assert b_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {ch}/{img}: {b_img.shape}"
            
            image[i, :, :] = b_img
        
        features = extract_features(image)
        
        Img_train.append(features)
        Label_train.append(mask_flatten)

    # Converter para arrays
    if len(ch_list) > 1:
        Img_train = np.vstack(Img_train) #empilhamento vertical, aumenta o numero de linhas (* em uma np array cada linha é um cjt de dados entre um [])
        Label_train = np.hstack(Label_train) #empilhamento horizontal, aumenta o numero de colunas (* em uma np array cada coluna é um numero dentro de um [])
    else:
        Img_train = np.array(Img_train)
        Label_train = np.array(Label_train)
        Img_train = Img_train.reshape(-1,1)
        Label_train = Label_train.reshape(-1)
        Label_train = Label_train.astype(np.uint8)
        
    # Escalar as características
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X_train = scaler.fit_transform(Img_train)
    if model == "MLP":
        #Scaler seletivo    
    
        # Cria matriz final
        X_train = np.zeros_like(Img_train, dtype=np.float32)
        
        # Escala apenas bandas
        scaler = MinMaxScaler(feature_range=(0, 1))
        if scale_cols:
            X_train[:, scale_cols] = scaler.fit_transform(Img_train[:, scale_cols])
        
        # Copia índices sem alterar
        if no_scale_cols:
            X_train[:, no_scale_cols] = Img_train[:, no_scale_cols]
            
        return X_train, Label_train, Imgs_test, y_test, scaler, Imgs_Selected
    else:
        X_train = Img_train
        
        return X_train, Label_train, Imgs_test, y_test, 0, Imgs_Selected
    


#versão inicial da função de testes
def test_routine_vs0(
        model,
        ch_url_dict,
        temp_folder,
        Imgs_test,
        ch_list,
        scaler_t,
        fold_nr,
        NS,
        plot_result=False,
        save_pred=True,
        MLE=True,
        mle_imgs_dir=None,
        WS=None,
        nr_classes=10):

    no_scale_channels = ['ndbi', 'ndvi', 'ndwi', 'ri']

    if MLE and mle_imgs_dir is None:
        raise ValueError("mle_imgs_dir deve ser definido quando MLE=True")

    # acumuladores de métricas
    iou_list = []
    skl_iou_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    bf_score_list = []

    # pasta de predições
    if save_pred:
        preds_folder = os.path.join(temp_folder, f'Preds_{fold_nr}')
        os.makedirs(preds_folder, exist_ok=True)

    for idx_img, img in enumerate(Imgs_test):

        label_path = os.path.join(ch_url_dict['Mask'], img)

        with rasterio.open(label_path) as src:
            mask = src.read(1)
            ref_meta = src.meta.copy()

        ref_meta.update({
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw"
        })

        for key in ["tiled", "blockxsize", "blockysize", "photometric", "interleave"]:
            ref_meta.pop(key, None)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask_bin = np.where(mask > 0, 1, 0)

        # empilhamento das bandas
        image = np.zeros((len(ch_list), NS[0], NS[1]), dtype=np.float32)

        for idx_ch, ch in enumerate(ch_list):

            img_path = os.path.join(ch_url_dict[ch], img)
            b_img = tiff.imread(img_path).astype(np.float32)

            assert b_img.shape == (NS[0], NS[1]), \
                f"Tamanho incompatível em canal {ch}, imagem {img}: {b_img.shape}"

            image[idx_ch] = b_img

        # extração de features
        Img_test = extract_features(image)

        # if len(ch_list) > 1:
        #     Img_test = np.vstack(Img_test)
        # else:
        #     Img_test = Img_test.reshape(-1, 1)

        # -------------------------------
        # normalização seletiva
        # -------------------------------

        n_features_per_channel = Img_test.shape[1] // len(ch_list)

        scale_cols = []
        no_scale_cols = []

        for idx_ch, ch in enumerate(ch_list):

            start = idx_ch * n_features_per_channel
            end = (idx_ch + 1) * n_features_per_channel

            if ch.lower() in no_scale_channels:
                no_scale_cols.extend(range(start, end))
            else:
                scale_cols.extend(range(start, end))

        Img_test_scaled = np.zeros_like(Img_test, dtype=np.float32)

        if scale_cols:
            Img_test_scaled[:, scale_cols] = scaler_t.transform(Img_test[:, scale_cols])

        if no_scale_cols:
            Img_test_scaled[:, no_scale_cols] = Img_test[:, no_scale_cols]

        Img_test = Img_test_scaled

        # -------------------------------
        # features estruturais (MLE)
        # -------------------------------

        if MLE:

            img_path = os.path.join(mle_imgs_dir, img)
            b_img = tiff.imread(img_path).astype(np.float32)

            assert b_img.shape == (NS[0], NS[1]), \
                f"Tamanho incompatível em MLE {img}: {b_img.shape}"

            features_dir = extract_features_directions_vs2(
                b_img,
                window_size=WS
            )

            X_test = np.hstack((Img_test, features_dir))

        else:

            X_test = Img_test

        # -------------------------------
        # predição
        # -------------------------------

        y_pred_np = model.predict(X_test).astype(np.uint8)#.flatten()
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        _, H, W = image.shape
        segmented_image = y_pred_np.reshape(H, W)

        # -------------------------------
        # métricas
        # -------------------------------

        iou, precision, recall, f1, _, _, _, _ = Calc_Metrics(
            segmented_image,
            mask_bin
        )

        bf_score = boundary_f1_score(mask_bin, segmented_image, tolerance=2)

        skl_mask_ref = skeletonize(mask_bin, method="zhang")
        skl_img_bin = skeletonize(segmented_image, method="zhang")

        skl_iou, _, _, _, _, _, _, _ = Calc_Metrics(
            skl_img_bin,
            skl_mask_ref
        )

        iou_list.append(iou)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        skl_iou_list.append(skl_iou)
        bf_score_list.append(bf_score)

        # -------------------------------
        # salvar predição
        # -------------------------------

        if save_pred:

            pred_path = os.path.join(preds_folder, img)

            with rasterio.open(pred_path, "w", **ref_meta) as dst:
                dst.write(segmented_image.astype(np.uint8), 1)

        # -------------------------------
        # visualização
        # -------------------------------

        if plot_result and idx_img < 20:

            prob_map = y_pred_proba.reshape(H, W)

            #base_cmap = plt.colormaps.get_cmap("viridis")
            #cmap = base_cmap.resampled(nr_classes)

            bins = np.linspace(0, 1, nr_classes + 1)

            prob_classes = np.digitize(prob_map, bins) - 1

            #norm = mcolors.BoundaryNorm(bins, cmap.N)

            salvar_resultados(
                image,
                mask,
                y_pred_np,
                prob_classes,
                H,
                W,
                img,
                temp_folder,
                bins,
                nr_classes
            )

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            image_rgb = np.transpose(image, (1, 2, 0))

            axes[0].imshow(
                image_rgb if image_rgb.ndim == 2 else image_rgb[:, :, 0],
                cmap='gray'
            )
            axes[0].set_title(f"Imagem Original_{img}")
            axes[0].axis("off")

            axes[1].imshow(mask_bin, cmap='gray')
            axes[1].set_title("Máscara Real")
            axes[1].axis("off")

            axes[2].imshow(segmented_image, cmap='gray')
            axes[2].set_title("Predição Binária")
            axes[2].axis("off")

            # im = axes[3].imshow(prob_classes, cmap=cmap, norm=norm)
            # axes[3].set_title("Mapa de Probabilidade")
            # axes[3].axis("off")

            fig.savefig(os.path.join(temp_folder, f"Teste_{img}.png"))
            plt.show()
            plt.close(fig)

    # -------------------------------
    # medianas finais
    # -------------------------------

    m_iou = np.median(iou_list)
    m_f1 = np.median(f1_list)
    m_rec = np.median(recall_list)
    m_prec = np.median(precision_list)
    m_bf = np.median(bf_score_list)
    m_skl_iou = np.median(skl_iou_list)

    return m_iou, m_f1, m_rec, m_prec, m_bf, m_skl_iou


def test_routine(
        model_t,
        model,
        ch_url_dict,
        preds_folder,
        Imgs_test,
        ch_list,
        scaler_t,
        fold_nr,
        NS,
        no_scale_channels,
        plot_result=False,
        save_pred=False,
        nr_classes=10):

    # acumuladores de métricas
    iou_list = []
    skl_iou_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    bf_score_list = []
    s_tp,s_fp,s_tn,s_fn = 0,0,0,0


    scale_cols = []
    no_scale_cols = []
    
    for idx, ch in enumerate(ch_list):
        if ch.lower() in no_scale_channels:
            no_scale_cols.append(idx)
        else:
            scale_cols.append(idx)

    for idx_img, img in enumerate(Imgs_test):

        label_path = os.path.join(ch_url_dict['Mask'], img)

        with rasterio.open(label_path) as src:
            mask = src.read(1)
            ref_meta = src.meta.copy()

        ref_meta.update({
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw"
        })

        for key in ["tiled", "blockxsize", "blockysize", "photometric", "interleave"]:
            ref_meta.pop(key, None)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask_bin = np.where(mask > 0, 1, 0)

        # empilhamento das bandas
        image = np.zeros((len(ch_list), NS[0], NS[1]), dtype=np.float32)

        for idx_ch, ch in enumerate(ch_list):

            img_path = os.path.join(ch_url_dict[ch], img)
            b_img = tiff.imread(img_path).astype(np.float32)

            assert b_img.shape == (NS[0], NS[1]), \
                f"Tamanho incompatível em canal {ch}, imagem {img}: {b_img.shape}"

            image[idx_ch] = b_img

        # extração de features
        Img_test = extract_features(image)

        if model_t == 'MLP':
            # -------------------------------
            # normalização seletiva
            # -------------------------------        
    
            # Cria matriz final
            Img_test_scaled = np.zeros_like(Img_test, dtype=np.float32)
            
            # Escala apenas bandas
    
            if scale_cols:
                Img_test_scaled[:, scale_cols] = scaler_t.transform(Img_test[:, scale_cols])
    
            if no_scale_cols:
                Img_test_scaled[:, no_scale_cols] = Img_test[:, no_scale_cols]
    
            X_test = Img_test_scaled
        else:
            X_test = Img_test

        # -------------------------------
        # predição
        # -------------------------------

        y_pred_np = model.predict(X_test).astype(np.uint8)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        _, H, W = image.shape
        segmented_image = y_pred_np.reshape(H, W)

        # -------------------------------
        # métricas
        # -------------------------------

        iou, precision, recall, f1, tp, fp, fn, tn = Calc_Metrics(
            segmented_image,
            mask_bin
        )


        bf_score = boundary_f1_score(mask_bin, segmented_image, tolerance=2)

        skl_mask_ref = skeletonize(mask_bin, method="zhang")
        skl_img_bin = skeletonize(segmented_image, method="zhang")

        skl_iou, _, _, _, _, _, _, _ = Calc_Metrics(
            skl_img_bin,
            skl_mask_ref
        )

        iou_list.append(iou)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        skl_iou_list.append(skl_iou)
        bf_score_list.append(bf_score)
        s_tp = s_tp + tp
        s_tn = s_tn + tn
        s_fp = s_fp + fp
        s_fn = s_fn + fn

        # -------------------------------
        # salvar predição
        # -------------------------------

        if save_pred:

            pred_path = os.path.join(preds_folder, img)

            with rasterio.open(pred_path, "w", **ref_meta) as dst:
                dst.write(segmented_image.astype(np.uint8), 1)

        # -------------------------------
        # visualização
        # -------------------------------

        if plot_result and idx_img < 20:
            
            temp_folder = os.path.join(preds_folder,'graphs')
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder, exist_ok=True)

            prob_map = y_pred_proba.reshape(H, W)

            #base_cmap = plt.colormaps.get_cmap("viridis")
            #cmap = base_cmap.resampled(nr_classes)

            bins = np.linspace(0, 1, nr_classes + 1)

            prob_classes = np.digitize(prob_map, bins) - 1

            #norm = mcolors.BoundaryNorm(bins, cmap.N)

            salvar_resultados(
                image,
                mask,
                y_pred_np,
                prob_classes,
                H,
                W,
                img,
                temp_folder,
                bins,
                nr_classes
            )

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            image_rgb = np.transpose(image, (1, 2, 0))

            axes[0].imshow(
                image_rgb if image_rgb.ndim == 2 else image_rgb[:, :, 0],
                cmap='gray'
            )
            axes[0].set_title(f"Imagem Original_{img}")
            axes[0].axis("off")

            axes[1].imshow(mask_bin, cmap='gray')
            axes[1].set_title("Máscara Real")
            axes[1].axis("off")

            axes[2].imshow(segmented_image, cmap='gray')
            axes[2].set_title("Predição Binária")
            axes[2].axis("off")

            # im = axes[3].imshow(prob_classes, cmap=cmap, norm=norm)
            # axes[3].set_title("Mapa de Probabilidade")
            # axes[3].axis("off")

            fig.savefig(os.path.join(temp_folder, f"Teste_{img}.png"))
            plt.show()
            plt.close(fig)

    # -------------------------------
    # medianas finais
    # -------------------------------

    m_iou = np.median(iou_list)
    m_f1 = np.median(f1_list)
    m_rec = np.median(recall_list)
    m_prec = np.median(precision_list)
    m_bf = np.median(bf_score_list)
    m_skl_iou = np.median(skl_iou_list)

    return m_iou, m_f1, m_rec, m_prec, m_bf, m_skl_iou, s_tp, s_fp, s_tn, s_fn


def TrainingLoop(model, 
                 ml_model, 
                 ch_url_dict, 
                 test_ch_url_dict,
                 test_imgs_dir, 
                 Imgs_List, 
                 Labels_List, 
                 train_idx, 
                 test_idx, 
                 N_total, 
                 NS, 
                 band_list, 
                 fold_idx,
                 r_masks,
                 preds_folder, 
                 no_scale_channels,
                 save_pred, 
                 plot_result,
                 nr_imgs_test=None):
    
    # # Listas para acumular métricas do CV
    # iou_list, f1_list, recall_list, precision_list, acc_list, bf_score_list = [], [], [], [], [], []

    # prepara dataset com características espectrais
    X_train, y_train, _, _, scaler_ds, Imgs_Selected = prepare_dataset(
        model,
        ch_url_dict,
        Imgs_List,
        Labels_List,
        train_idx=train_idx,
        n_total=N_total,
        NS=NS,
        ch_list=band_list,
        fold_id=fold_idx,
        no_scale_channels=no_scale_channels,
        t_size=None,
        global_seed=42
    )
    

    #-------- TREINAMENTO -------    
    if model == "MLP":
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train)
        
        print('- Treinando o modelo....')
        ml_model.fit(
            X_train,
            y_train.ravel(),
            sample_weight=sample_weights
        )
    else:
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train)
        
        print('- Treinando o modelo....')
        ml_model.fit(
            X_train,
            y_train.ravel(),
            sample_weight=sample_weights
        )

        # ml_model.fit(X_train, y_train.ravel())

    # if model == "MLP":
    #     # Curva de erro
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(ml_model.loss_curve_)
    #     plt.title('Curva de erro durante o treinamento')
    #     plt.xlabel('Época')
    #     plt.ylabel('Erro (Loss)')
    #     plt.grid(True)
    #     plt.show()
        
    #     total_params = sum(w.size for w in ml_model.coefs_) + \
    #            sum(b.size for b in ml_model.intercepts_)

    #     print(f"Nr Parametrs MLP: {total_params}")

    # -------- TESTE DO FOLD -------
    if r_masks:
        Imgs_test = [ f for f in os.listdir(os.path.join(test_imgs_dir, band_list[0])) if f.lower().endswith(('.tif', '.tiff'))]
        if nr_imgs_test:
            Imgs_test = random.Random(42).sample(Imgs_test, nr_imgs_test)
        
    else:
        TestImgsList = os.listdir(os.path.join(test_imgs_dir, band_list[0]))
        Imgs_test = [TestImgsList[i] for i in test_idx]
    
    # ------------------------------------------------------------------    
    print('- Testando o modelo...')

    m_iou, m_f1, m_rec, m_prec, m_bf, m_skl_iou, s_tp, s_fp, s_tn, s_fn = test_routine(
        model,
        model=ml_model,
        ch_url_dict=test_ch_url_dict,
        preds_folder = preds_folder,
        Imgs_test=Imgs_test,
        ch_list=band_list,
        scaler_t=scaler_ds,
        fold_nr=fold_idx + 1,
        NS=NS,
        no_scale_channels=no_scale_channels,
        plot_result=plot_result,
        save_pred=save_pred)
                
    return m_iou, m_f1, m_rec, m_prec, m_bf, m_skl_iou, s_tp, s_fp, s_tn, s_fn, ml_model, scaler_ds, Imgs_test

#implementa a filtragem espacial 
def spatial_filtering_vs0(img_p, img_f):

    # garante binário
    img_p = img_p > 0
    img_f = img_f > 0

    # rotula componentes conectados
    labels = skm.label(img_p, connectivity=2)

    # imagem de saída
    output = np.zeros_like(img_p, dtype=np.uint8)

    # percorre componentes
    for lab in range(1, labels.max() + 1):

        component = labels == lab

        # verifica interseção
        if np.any(component & img_f):
            output[component] = 1

    return output

def spatial_filtering_vs1(img_p, img_f):

    # garante binário
    img_p = img_p > 0
    img_f = img_f > 0

    # rotula componentes conectados
    labels = skm.label(img_p, connectivity=2)

    # imagem de saída
    output = np.zeros_like(img_p, dtype=np.uint8)

    # percorre componentes
    for lab in range(1, labels.max() + 1):

        component = labels == lab

        # pixels de interseção
        intersection = component & img_f

        # se houver interseção
        if np.any(intersection):

            # adiciona o componente de img_p
            output[component] = 1

            # adiciona os pixels de img_f que intersectam
            output[intersection] = 1

    return output

def spatial_filtering_vs2(img_p, img_f):

    # garante binário
    img_p = img_p > 0
    img_f = img_f > 0

    # -------------------------------------------------
    # Componentes de img_p
    # -------------------------------------------------
    labels_p = skm.label(img_p, connectivity=2)

    output = np.zeros_like(img_p, dtype=np.uint8)

    for lab in range(1, labels_p.max() + 1):

        component_p = labels_p == lab

        # interseção com img_f
        if np.any(component_p & img_f):
            output[component_p] = 1

    # -------------------------------------------------
    # Componentes de img_f
    # -------------------------------------------------
    labels_f = skm.label(img_f, connectivity=2)

    for lab in range(1, labels_f.max() + 1):

        component_f = labels_f == lab

        # interseção com img_p
        if np.any(component_f & output):
            output[component_f] = 1

    return output

 
def MLE_Data_Filering(model,
                      ml_model,
                      ch_url_dict,
                      band_list,
                      mle_folder, 
                      label_dir,
                      img_list, 
                      scaler_t, 
                      output_f, 
                      NS,
                      no_scale_channels,
                      save_results=True, 
                      show_imgs=False):
    
    metrics_hist = {
        "img_nm":[],
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "skl_iou": [],
        "skl_precision": [],
        "skl_recall": [],
        "skl_f1": [],
        "bf_score": [],
        "tp":[],
        "fp":[],
        "tn":[],
        "fn":[]
        }
    
    metrics_hist_si = {
        "img_nm":[],
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "skl_iou": [],
        "skl_precision": [],
        "skl_recall": [],
        "skl_f1": [],
        "bf_score": [],
        "tp":[],
        "fp":[],
        "tn":[],
        "fn":[]
        }
        
    scale_cols = []
    no_scale_cols = []
    
    for idx, ch in enumerate(band_list):
        if ch.lower() in no_scale_channels:
            no_scale_cols.append(idx)
        else:
            scale_cols.append(idx)
    
    for idx, img in enumerate(img_list):
        # print(f"Imagem: {img}")
        # print("--------------------------------------------------------------")

        label_path = os.path.join(label_dir, img)
   
        with rasterio.open(label_path) as src:
            mask = src.read(1)
            ref_meta = src.meta.copy()
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        mask_bin = np.where(mask > 0, 1, 0)
        
        # plt.title(f"Mascara {img}")
        # plt.imshow(mask_bin, cmap="gray")
        # plt.axis("off")
        # plt.show()
    
        # Cria as imagens empilhadas
        image = np.zeros((len(band_list), NS[0], NS[1]), dtype=np.float32)
        
        for i, ch in enumerate(band_list):
            img_path = os.path.join(ch_url_dict[ch], img)
            b_img = tiff.imread(img_path).astype(np.float32)
            
            if b_img.ndim == 3:
                b_img = b_img[..., 0]
            
            # Verifica tamanho
            assert b_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {ch}/{img}: {b_img.shape}"
            
            image[i, :, :] = b_img
            
        #Extract Features
        H, W = mask_bin.shape
        # Label_test = mask_bin.ravel().astype(np.uint8)

        # --- Extração de características ---
        Img_test = extract_features(image)

        # # Normalização
        # Img_test = scaler_t.transform(Img_test)
        
        if model == 'MLP':
            # -------------------------------
            # Normalização seletiva
            # -------------------------------
            
            # Cria matriz final
            Img_test_scaled = np.zeros_like(Img_test, dtype=np.float32)
            
            # Aplica scaler apenas onde necessário
            if scale_cols:
                Img_test_scaled[:, scale_cols] = scaler_t.transform(Img_test[:, scale_cols])
            
            # Mantém índices inalterados
            if no_scale_cols:
                Img_test_scaled[:, no_scale_cols] = Img_test[:, no_scale_cols]
            
            X_test = Img_test_scaled
        else:
            X_test = Img_test
            
        #Segmentação da imagem
        y_pred_np = ml_model.predict(X_test).astype(np.uint8)
        
        #Recupera características estruturais
        img_path = os.path.join(mle_folder, img)
        mle_img = tiff.imread(img_path).astype(np.uint16)

        assert mle_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {img}: {mle_img.shape}"
        
        # plt.title(f"MLE {img}")
        # plt.imshow(mle_img, cmap="gray")
        # plt.axis("off")
        # plt.show()
        
        #redimensiona as predições
        segmented_image = y_pred_np.reshape(H, W)
        
        # plt.title(f"segmented_image {img}")
        # plt.imshow(segmented_image, cmap="gray")
        # plt.axis("off")
        # plt.show()
        
        #Filtragem por modelo de ML
        img_f = ml_filter_mask_vs2(ml_model,
                               X_test,
                               mle_img,
                               H,
                               W,
                               prob_thr=0.5,
                               use_proba=True)

        # plt.title(f"Imagem filtrada 1 {img}")
        # plt.imshow(img_f*255, cmap="gray")
        # plt.axis("off")
        # plt.show()
        
        #==========================================================
        # #abre a imagem utilizada no MLE        
        # img_path = os.path.join(mle_band_dir, img)
        # with rasterio.open(img_path) as src:
        #     mle_img = src.read(1)
        #     meta = src.meta.copy()
        
        # #aplica a dilatação condicional
        # rec_img = ConditionalDilation(img_f, mle_img)
        
        # plt.title(f"Imagem Reconstruída 1{img}")
        # plt.imshow(rec_img*255, cmap="gray")
        # plt.axis("off")
        # plt.show()
        
        #Aplica a filtragem espacial por esqueletos
        rec_img = spatial_filtering_vs2(segmented_image, img_f)
        
        # plt.title(f"Imagem Final {img}")
        # plt.imshow(rec_img*255, cmap="gray")
        # plt.axis("off")
        # plt.show()
        
        # # Imagem 1
        # plt.subplot(1, 3, 1)
        # plt.imshow(mask_bin, cmap='gray')
        # plt.title('mask_bin')
        # plt.axis('off')
        
        # # Imagem 2
        # plt.subplot(1, 3, 2)
        # plt.imshow(segmented_image, cmap='gray')
        # plt.title('segmented_image')
        # plt.axis('off')
        
        # # Imagem 3
        # plt.subplot(1, 3, 3)
        # plt.imshow(rec_img, cmap='gray')
        # plt.title('rec_img')
        # plt.axis('off')
        
        # plt.tight_layout()
        # plt.show()
        
        # ask_del = input(f'Resultados Imagem {img}. Deseja prosseguir com a análise (y, n)?')
        # if ask_del == 'n':
        #     break
        
        #########################################################
        img_bin = (rec_img > 0).astype(np.uint8)

        # imagem de saída (uint8) APENAS para salvamento
        img_out = (img_bin * 255)

        # --------------------------------------------------
        # Salvamento com georreferenciamento
        # --------------------------------------------------
        if save_results:
            
            meta_out = ref_meta.copy()
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
                os.path.join(output_f, img), "w", **meta_out
            ) as dst:
                dst.write(img_out, 1)

        # Métricas imagem segmentada
        (iou_si, precision_si, recall_si, f1_si, tp_si, fp_si, fn_si, tn_si) = Calc_Metrics(segmented_image, mask_bin)
        
        #bf-score
        bf_score_si = boundary_f1_score(mask_bin, segmented_image, tolerance=2)

        metrics_hist_si["img_nm"].append(img)
        metrics_hist_si["iou"].append(iou_si)
        metrics_hist_si["f1"].append(f1_si)
        metrics_hist_si["precision"].append(precision_si)
        metrics_hist_si["recall"].append(recall_si)
        metrics_hist_si["bf_score"].append(bf_score_si)
        metrics_hist_si["tp"].append(tp_si)
        metrics_hist_si["fp"].append(fp_si)
        metrics_hist_si["tn"].append(tn_si)
        metrics_hist_si["fn"].append(fn_si)
        
        # --------------------------------------------------
        # Métricas
        # --------------------------------------------------
        (iou, precision, recall, f1, tp, fp, fn, tn) = Calc_Metrics(img_bin, mask_bin)

        #bf-score
        bf_score = boundary_f1_score(mask_bin, img_bin, tolerance=2)

        # --------------------------------------------------
        # Métricas sobre esqueletos
        # --------------------------------------------------

        skl_mask_ref = skeletonize(mask_bin, method="zhang")
        skl_img_bin = skeletonize(img_bin, method="zhang")

        # if img_bin.sum() == 0:
        #     print(f"[WARN] Esqueleto previsto vazio: {f}")
        
        # if skl_mask_ref.sum() == 0:
        #     print(f"[WARN] Esqueleto GT vazio: {f}")
            
        (skl_iou,_, _, _, _, _, _, _) = Calc_Metrics(skl_img_bin, skl_mask_ref)

        metrics_hist["img_nm"].append(img)
        metrics_hist["iou"].append(iou)
        metrics_hist["f1"].append(f1)
        metrics_hist["precision"].append(precision)
        metrics_hist["recall"].append(recall)
        metrics_hist["skl_iou"].append(skl_iou)
        metrics_hist["bf_score"].append(bf_score)
        metrics_hist["tp"].append(tp)
        metrics_hist["fp"].append(fp)
        metrics_hist["tn"].append(tn)
        metrics_hist["fn"].append(fn)

        #metrics_hist["median_length_skl"].append(img_skl_stats["median_length"])

        # #imprime as métricas da imagem segmentada e da imagem resultante do MLE
        # print("=====================================")
        # print("Metricas MLE Image x Segmented Image")
        # print("=====================================")
        # print(f"IoU: {iou} x {iou_si}")
        # print(f"F1-Score: {f1} x {f1_si}")
        # print(f"Recall: {recall} x {recall_si}")
        # print(f"Precision: {precision} x {precision_si}")
        # print("=====================================")
        
        # --------------------------------------------------
        # Visualização opcional
        # --------------------------------------------------
        if show_imgs:
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 3, 1)
            plt.title(f"Mascara {img}")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title(f"Linear enhancement - {img}")
            plt.imshow(img_out, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Segmented Image - {img}")
            plt.imshow(segmented_image, cmap="gray")
            plt.axis("off")
            
            plt.tight_layout()
            
            if save_results:
                output_f = os.path.join(output_f,'Figs')
                os.makedirs(output_f, exist_ok=True)
                plt.savefig(os.path.join(output_f, f"Comparativo - Img {img}.png"), dpi=300, bbox_inches="tight")

            plt.show()
            
    return metrics_hist, metrics_hist_si

def ml_filter_mask_vs0(model,
                   features,
                   MLE_Img,
                   H,
                   W,
                   prob_thr=0.5,
                   use_proba=True):

    # -------------------------------------------------
    # Garantia de formato 2D
    # -------------------------------------------------
    if MLE_Img.ndim != 2:
        raise ValueError("MLE_Img deve ser 2D")

    if MLE_Img.shape != (H, W):
        raise ValueError("Dimensão inconsistente entre MLE_Img e (H,W)")

    # -------------------------------------------------
    # Índices ativos (2D)
    # -------------------------------------------------
    active_mask = (MLE_Img > 0)

    n_active = np.sum(active_mask)

    if n_active == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # -------------------------------------------------
    # Seleção de features
    # features é (H*W, N)
    # -------------------------------------------------
    features_2d = features.reshape(H, W, -1)
    features_sel = features_2d[active_mask]

    # -------------------------------------------------
    # Predição
    # -------------------------------------------------
    if use_proba and hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_sel)[:, 1]
        keep = (probs >= prob_thr).astype(np.uint8)
    else:
        keep = model.predict(features_sel).astype(np.uint8)

    # -------------------------------------------------
    # Reconstrução direta 2D
    # -------------------------------------------------
    filtered_mask = np.zeros((H, W), dtype=np.uint8)
    filtered_mask[active_mask] = keep

    return filtered_mask

def ml_filter_mask_vs1(model,
                      features,
                      MLE_Img,
                      H,
                      W,
                      prob_thr=0.5,
                      use_proba=True):

    # -------------------------------------------------
    # Garantia de formato 2D
    # -------------------------------------------------
    if MLE_Img.ndim != 2:
        raise ValueError("MLE_Img deve ser 2D")

    if MLE_Img.shape != (H, W):
        raise ValueError("Dimensão inconsistente entre MLE_Img e (H,W)")

    # -------------------------------------------------
    # Índices ativos (domínio MLE)
    # -------------------------------------------------
    active_mask = (MLE_Img > 0)

    if not np.any(active_mask):
        return np.zeros((H, W), dtype=np.uint8)

    # -------------------------------------------------
    # Seleção de features
    # -------------------------------------------------
    features_2d = features.reshape(H, W, -1)
    features_sel = features_2d[active_mask]

    # -------------------------------------------------
    # Predição (seed)
    # -------------------------------------------------
    if use_proba and hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_sel)[:, 1]
        keep = (probs >= prob_thr)
    else:
        keep = model.predict(features_sel).astype(bool)

    # -------------------------------------------------
    # Reconstrução da seed em 2D
    # -------------------------------------------------
    seed = np.zeros((H, W), dtype=bool)
    seed[active_mask] = keep

    # -------------------------------------------------
    # 🔴 NOVA ETAPA: componentes conexos da MLE
    # -------------------------------------------------
    structure = np.ones((3, 3), dtype=np.int8)  # conectividade 8
    labeled, _ = ndi.label(active_mask, structure=structure)

    # rótulos que intersectam a seed
    labels_to_keep = np.unique(labeled[seed])
    labels_to_keep = labels_to_keep[labels_to_keep != 0]

    # -------------------------------------------------
    # Máscara final: componentes completos
    # -------------------------------------------------
    final_mask = np.isin(labeled, labels_to_keep)

    return final_mask.astype(np.uint8)


def ml_filter_mask_vs2(model,
                      features,
                      MLE_Img,
                      H,
                      W,
                      prob_thr=0.5,
                      comp_ratio_thr=0.1,   # fração mínima de pixels positivos
                      comp_min_size=20,     # tamanho mínimo do componente
                      use_proba=True):

    # -------------------------------------------------
    # Validação
    # -------------------------------------------------
    if MLE_Img.ndim != 2:
        raise ValueError("MLE_Img deve ser 2D")

    if MLE_Img.shape != (H, W):
        raise ValueError("Dimensão inconsistente entre MLE_Img e (H,W)")

    # -------------------------------------------------
    # Domínio ativo (MLE)
    # -------------------------------------------------
    active_mask = (MLE_Img > 0)

    if not np.any(active_mask):
        return np.zeros((H, W), dtype=np.uint8)

    # -------------------------------------------------
    # Features
    # -------------------------------------------------
    features_2d = features.reshape(H, W, -1)
    features_sel = features_2d[active_mask]

    # -------------------------------------------------
    # Predição
    # -------------------------------------------------
    if use_proba and hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_sel)[:, 1]
        keep = (probs >= prob_thr)
    else:
        probs = None
        keep = model.predict(features_sel).astype(bool)

    # -------------------------------------------------
    # Reconstrução 2D
    # -------------------------------------------------
    seed = np.zeros((H, W), dtype=bool)
    seed[active_mask] = keep

    if probs is not None:
        prob_map = np.zeros((H, W), dtype=np.float32)
        prob_map[active_mask] = probs
    else:
        prob_map = None

    # -------------------------------------------------
    # Componentes conexos da MLE
    # -------------------------------------------------
    structure = np.ones((3, 3), dtype=np.int8)
    labeled, num_labels = ndi.label(active_mask, structure=structure)

    output = np.zeros((H, W), dtype=np.uint8)

    # -------------------------------------------------
    # Avaliação por componente
    # -------------------------------------------------
    for lab in range(1, num_labels + 1):

        component = (labeled == lab)
        comp_size = np.sum(component)

        # descarta componentes muito pequenos
        if comp_size < comp_min_size:
            continue

        # pixels positivos dentro do componente
        n_pos = np.sum(seed & component)

        if n_pos == 0:
            continue

        # proporção de positivos
        ratio = n_pos / comp_size

        # -------------------------------------------------
        # Critérios de decisão
        # -------------------------------------------------
        keep_component = False

        # critério 1: proporção mínima
        if ratio >= comp_ratio_thr:
            keep_component = True

        # critério 2 (opcional): probabilidade média
        if prob_map is not None:
            mean_prob = np.mean(prob_map[component])
            
            # combinação (ajustável)
            if mean_prob > prob_thr and ratio > (comp_ratio_thr * 0.5):
                keep_component = True

        # -------------------------------------------------
        if keep_component:
            output[component] = 1

    return output

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

    # --- Normalização ---
    img_norm = img_1ch.astype(np.float32)
    img_norm -= img_norm.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()
    #img_norm = img_1ch
    
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


def ConditionalDilation_vs2(
    mask,
    img_1ch,
    m_dist=3,
    pctls=(25, 90),
    spatial_lim=True,
    spectral_lim=True
):

    # Marcador binário
    marker_bin = (mask > 0).astype(np.uint8)

    if marker_bin.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # --- Normalização ---
    img_norm = img_1ch.astype(np.float32)
    img_norm -= img_norm.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()

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
    # Domínio binário permitido
    # --------------------------------------------------
    domain_mask = spatial_mask & spectral_mask

    # Garantia: marcador deve estar contido no domínio
    marker_bin = marker_bin & domain_mask

    # --------------------------------------------------
    # Reconstrução binária
    # --------------------------------------------------
    rec = reconstruction(
        seed=marker_bin.astype(np.uint8),
        mask=domain_mask.astype(np.uint8),
        method='dilation'
    )

    return (rec > 0).astype(np.uint8)


def generate_band_combinations(features, min_k=2, max_k=None):
    if max_k is None:
        max_k = len(features)
        
    comb_list = []
    for k in range(min_k, max_k+1):
        comb_list.extend(list(combinations(features, k)))
        
    return comb_list


def evaluate_band_combination(model, 
                              band_list_current,
                              Imgs_List,
                              Labels_List,
                              trn_imgs_dir,
                              trn_mask_dir,
                              test_imgs_dir,
                              test_mask_dir,
                              k_fold,
                              NS,
                              N_total,
                              r_masks,
                              output_path,
                              no_scale_channels, 
                              h_layers = None,
                              nr_imgs_test = None):

    print(f"\n\n========== TESTANDO BANDAS: {band_list_current} ==========")

    # IMG_CHANNELS = len(band_list_current)

    # Atualiza dicionários de canais
    ch_url_dict = {band: os.path.join(trn_imgs_dir, band) 
                   for band in band_list_current}
    ch_url_dict['Mask'] = trn_mask_dir

    test_ch_url_dict = {band: os.path.join(test_imgs_dir, band) 
                        for band in band_list_current}
    test_ch_url_dict['Mask'] = test_mask_dir

    iou_all_folds, f1_all_folds, rec_all_folds, prec_all_folds, bf_all_folds = [], [], [], [], []

    kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    for fold_idx, (train_rel, test_rel) in enumerate(kfold.split(Imgs_List)):
        print(f"\n========== Fold {fold_idx + 1}/{k_fold} ==========")
        start_time = time.time()

        train_idx = np.array(train_rel)
        test_idx  = np.array(test_rel)
        
        if model == 'MLP':
            ml_model = MLPClassifier(
                hidden_layer_sizes = h_layers,#(10,20,10,2), #(2, 6, 18, 32, 18, 6, 2), ######(64,128,256)#####(16, 32,64,128, 256, 128, 64, 32,16) #(8,16,32,64,32,16,8)
                activation='relu',
                solver='adam',
                max_iter=200,
                verbose=True,
                early_stopping=True)
            
        elif model == 'RF':
            
            ml_model = RandomForestClassifier(
                n_estimators=100,
                criterion = "gini",
                min_samples_split = 2,
                min_samples_leaf = 2,
                max_features = 'sqrt',
                bootstrap = True,
                max_leaf_nodes = 10,
                n_jobs = -1,
                class_weight="balanced"
                )
        

        iou_list, f1_list, recall_list, precision_list, bf_score_list, skl_iou_list, *_ = TrainingLoop(
            model,
            ml_model,
            ch_url_dict,
            test_ch_url_dict,
            test_imgs_dir,
            Imgs_List,
            Labels_List,
            train_idx,
            test_idx,
            N_total,
            NS,
            band_list_current,
            fold_idx,
            r_masks,
            output_path,
            no_scale_channels,
            save_pred=False,
            plot_result=False,
            nr_imgs_test=nr_imgs_test
            )

        iou_all_folds.append(iou_list)
        f1_all_folds.append(f1_list)
        rec_all_folds.append(recall_list)
        prec_all_folds.append(precision_list)
        bf_all_folds.append(bf_score_list)
        
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"               - Tempo do treinamento: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    median_iou = np.median(iou_all_folds)
    std_iou  = np.std(iou_all_folds)
    median_f1 = np.median(f1_all_folds)
    std_f1  = np.std(f1_all_folds)
    median_rec = np.median(rec_all_folds)
    std_rec  = np.std(rec_all_folds)
    median_prec = np.median(prec_all_folds)
    std_prec  = np.std(prec_all_folds)
    median_bf = np.median(bf_all_folds)
    std_bf  = np.std(bf_all_folds)

    return median_iou, std_iou,median_f1, std_f1, median_rec, std_rec, median_prec, std_prec, median_bf, std_bf
