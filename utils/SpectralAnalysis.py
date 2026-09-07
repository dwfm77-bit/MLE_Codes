#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 10:30:06 2026

@author: dwander
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import rasterio
from tqdm import tqdm
from rasterio.enums import Compression
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from skimage.morphology import skeletonize
import itertools
import pandas as pd

# ======================= DIRETÓRIOS ===========================
DS_DIR = '/home/usuario/Documentos/1.Doutorado/Datasets/S2-RD'
TRN_DIR = os.path.join(DS_DIR, 'Trn')

TEMP_FOLDER = '/home/usuario/Documentos/1.Doutorado/Testes//Testes_Clustering/Analise_Espectral_Multibanda'
os.makedirs(TEMP_FOLDER, exist_ok=True)

show_imgs=True
# ======================= CANAIS ===============================
channel_list = ['b8', 'b11', 'b12', 'ndvi', 'ndwi', 'ri'] #['b2', 'b3', 'b4', 'b8', 'b11', 'b12', 'ndbi', 'ndvi', 'ndwi', 'ri']

# ======================= GMM PARAMS ===========================
gmm_params = dict(
    n_components=1,
    covariance_type='full',
    init_params='kmeans',
    max_iter=200,
    random_state=42
)

# ======================= MÉTRICAS =============================
def iou_np(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return inter / (union + 1e-8)

def precision_np(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    return tp / (tp + fp + 1e-8)

def recall_np(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    return tp / (tp + fn + 1e-8)

def skeletonize_mask(mask, im_show=True):
    # Normaliza para booleano
    mask_bool = mask > 0
    skel = skeletonize(mask_bool,method='zhang')

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


def save_prediction_geotiff(
    ref_img_path,
    out_path,
    pred_mask
):
    """
    Salva máscara binária preservando georreferenciamento da imagem de referência.
    """
    with rasterio.open(ref_img_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw',
        nodata=0
    )

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(pred_mask.astype(np.uint8), 1)
# ==============================================================
# ========== FASE 1 — ANÁLISE ESPECTRAL DAS ESTRADAS ============
# ==============================================================

road_pixels = defaultdict(list)

image_files = os.listdir(os.path.join(TRN_DIR, 'Imgs', channel_list[0]))
image_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff'))]

print('\n[1] Coletando pixels de estrada por banda...\n')

for ch in channel_list:
    print(f'Banda: {ch}')
    for img_name in image_files:

        img_path = os.path.join(TRN_DIR, 'Imgs', ch, img_name)
        mask_path = os.path.join(TRN_DIR, 'Mask', img_name)

        img = tiff.imread(img_path).astype(np.float32)
        
        if img.max() > 1:
            #normaliza a imagem
            img_norm = (img - img.min()) / (img.max() - img.min())
        else:
            img_norm = img

        
        mask = (tiff.imread(mask_path) > 0).astype(np.uint8)
        
        if mask.ndim == 3:
            mask = mask[:,:,0]

        skl_mask = skeletonize_mask(mask, im_show=False)

        vals = img_norm[skl_mask > 0]

        if vals.size > 0:
            road_pixels[ch].extend(vals.tolist())

# ======================= INTERVALOS ===========================
percentis_dict = {}

for ch, vals in road_pixels.items():
    vals = np.array(vals)

    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])

    percentis_dict[ch] = {
        'low': p10,
        'high': p90,
        'p25': p25,
        'p50': p50,
        'p75': p75
    }

# ======================= BOXPLOTS =============================

plt.figure(figsize=(15, 6))

data = [road_pixels[ch] for ch in channel_list]

plt.boxplot(
    data,
    labels=channel_list,
    showfliers=True,
    notch=True
)

for i, ch in enumerate(channel_list, start=1):
    low = percentis_dict[ch]['low']
    high = percentis_dict[ch]['high']
    plt.plot([i-0.25, i+0.25], [low, low], 'r-')
    plt.plot([i-0.25, i+0.25], [high, high], 'r-')

plt.ylabel('Nível de cinza / reflectância')
plt.title('Distribuição espectral dos pixels de estrada por banda')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(os.path.join(TEMP_FOLDER, 'boxplot_espectral_estradas.png'), dpi=300)
plt.show()

# ==============================================================
# ========== FASE 2 — GMM GUIADO PELOS INTERVALOS ===============
# ==============================================================

print('\n[2] Treinando e avaliando GMM por combinação de 3 bandas...\n')

results = []

# Todas as combinações de 3 bandas
band_combinations = list(itertools.combinations(channel_list, 3))

for bands in tqdm(band_combinations, desc=" Teste de Combinação de bandas"):

    lows  = [percentis_dict[b]['low']  for b in bands]
    highs = [percentis_dict[b]['high'] for b in bands]

    for img_name in image_files:

        imgs = []
        valid = True

        # --- leitura e normalização ---
        for b in bands:
            img_path = os.path.join(TRN_DIR, 'Imgs', b, img_name)
            img = tiff.imread(img_path).astype(np.float32)

            if img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            imgs.append(img)

        imgs = np.stack(imgs, axis=-1)  # (H, W, 3)

        # máscara GT
        mask_path = os.path.join(TRN_DIR, 'Mask', img_name)
        mask = (tiff.imread(mask_path) > 0).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # --- seleção por intervalo conjunto ---
        sel = np.ones(imgs.shape[:2], dtype=bool)
        for i in range(3):
            sel &= (imgs[:, :, i] >= lows[i]) & (imgs[:, :, i] <= highs[i])

        X = imgs[sel]  # shape (N, 3)

        if X.shape[0] < 20:
            continue

        # --- GMM ---
        gmm = GaussianMixture(**gmm_params)
        gmm.fit(X)

        logp_train = gmm.score_samples(X)
        threshold = np.percentile(logp_train, 75)

        # --- aplicação na imagem inteira ---
        X_all = imgs.reshape(-1, 3)
        logp = gmm.score_samples(X_all)

        y_pred = (logp >= threshold).reshape(mask.shape)

        # --- métricas ---
        iou  = iou_np(y_pred, mask)
        prec = precision_np(y_pred, mask)
        rec  = recall_np(y_pred, mask)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        
        # --- salvamento da predição ---        

        results.append({
            'bands': '_'.join(bands),
            'image': img_name,
            'iou': iou,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'n_samples': X.shape[0]
        })

        
df = pd.DataFrame(results)

# métricas médias por combinação
summary = (
    df.groupby('bands')
      .agg(
          mean_iou=('iou', 'mean'),
          mean_f1=('f1', 'mean'),
          mean_precision=('precision', 'mean'),
          mean_recall=('recall', 'mean'),
          n_images=('image', 'count'),
          mean_samples=('n_samples', 'mean')
      )
      .reset_index()
)


out_csv = os.path.join(TEMP_FOLDER, 'gmm_triplets_summary.csv')
summary.to_csv(out_csv, index=False)

print(f'\nResumo salvo em: {out_csv}')


#Identifica a melhor combinação de bandas em termos de IoU
best_f1 = summary.sort_values(
    by='mean_f1',
    ascending=False
).iloc[0]

best_ch = best_f1.iloc[0]

best_bands = best_ch.split("_")

out_path = os.path.join(TEMP_FOLDER, best_ch)
os.makedirs(out_path, exist_ok=True)

#salva as imagens da melhor combinação
for img_name in image_files:

    imgs = []
    valid = True

    # --- leitura e normalização ---
    for b in best_bands:
        img_path = os.path.join(TRN_DIR, 'Imgs', b, img_name)
        img = tiff.imread(img_path).astype(np.float32)

        if img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        imgs.append(img)

    imgs = np.stack(imgs, axis=-1)  # (H, W, 3)

    # máscara GT
    mask_path = os.path.join(TRN_DIR, 'Mask', img_name)
    mask = (tiff.imread(mask_path) > 0).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # --- seleção por intervalo conjunto ---
    sel = np.ones(imgs.shape[:2], dtype=bool)
    for i in range(3):
        sel &= (imgs[:, :, i] >= lows[i]) & (imgs[:, :, i] <= highs[i])

    X = imgs[sel]  # shape (N, 3)

    if X.shape[0] < 20:
        continue

    # --- GMM ---
    gmm = GaussianMixture(**gmm_params)
    gmm.fit(X)

    logp_train = gmm.score_samples(X)
    threshold = np.percentile(logp_train, 60)

    # --- aplicação na imagem inteira ---
    X_all = imgs.reshape(-1, 3)
    logp = gmm.score_samples(X_all)

    y_pred = (logp >= threshold).reshape(mask.shape)
    
    # --- salvamento da predição ---        

    # usa a primeira banda como referência geoespacial
    ref_img_path = os.path.join(TRN_DIR, 'Imgs', b, img_name)
    
    output_file = os.path.join(out_path, img_name)
    
    save_prediction_geotiff(
        ref_img_path=ref_img_path,
        out_path=output_file,
        pred_mask=y_pred
    )

    if show_imgs:
        # --- visualização ---
        plt.figure(figsize=(10, 4))
    
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {img_name}')
        plt.axis('off')
    
        plt.subplot(1, 2, 2)
        plt.imshow(y_pred, cmap='gray')
        plt.title(f'GMM {img_name}')
        plt.axis('off')
    
        plt.tight_layout()
        plt.show()