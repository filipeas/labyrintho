import tifffile
import numpy as np
import cv2
import random
import os

def main():
    input_path = '/Users/filipealvessampaio/Documents/workspace/labyrintho/tmp/s0/inline_100.tiff'
    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)

    # Lê a imagem TIFF mantendo os dados exatamente como estão
    img = tifffile.imread(input_path)

    # Se for imagem 2D (1 canal), converte para 3 canais replicando
    if img.ndim == 2:
        img_rgb = np.stack([img]*3, axis=-1)  # (H, W) → (H, W, 3)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_rgb = np.repeat(img, 3, axis=2)  # (H, W, 1) → (H, W, 3)
    else:
        img_rgb = img  # Já é multicanal

    # Rotacionar -90 graus (sentido anti-horário)
    img_rgb = np.rot90(img_rgb, k=1)  # k=1 equivale a -90°

    h, w, _ = img_rgb.shape
    crop_h, crop_w = 1006, 590

    # Garante que a região vai caber
    assert h >= crop_h and w >= crop_w, f"Imagem pequena demais para crop ({h}, {w}) < ({crop_h}, {crop_w})"

    # Escolhe coordenadas aleatórias
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)

    # Recorta a região
    cropped = img_rgb[y:y+crop_h, x:x+crop_w]

    # Desenha contorno na imagem original
    img_rgb_boxed = img_rgb.copy()
    cv2.rectangle(img_rgb_boxed, (x, y), (x + crop_w, y + crop_h), color=(0, 0, 255), thickness=2)

    # Salva imagens
    tifffile.imwrite(os.path.join(output_folder, 'recorte.tiff'), cropped)
    cv2.imwrite(os.path.join(output_folder, 'com_regiao.png'), img_rgb_boxed)

    print(f"Imagem original (com contorno): {img_rgb_boxed.shape}")
    print(f"Recorte salvo: {cropped.shape}")
    print(f"Coordenadas do recorte: x={x}, y={y}, largura={crop_w}, altura={crop_h}")

if __name__ == "__main__":
    main()