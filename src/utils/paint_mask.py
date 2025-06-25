import numpy as np
import argparse
import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def paint_mask(path: str, filename: str, only_iou: str):
    print(path + filename)
    img = cv2.imread(path + filename, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Erro: Não foi possível carregar a imagem.")
        return

    # if len(img.shape) > 2 and img.shape[2] == 3:
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # elif len(img.shape) > 2 and img.shape[2] == 4:
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # else:
    img_gray = img

    print("Níveis de cinza detectados: ", np.unique(img_gray))

    if only_iou == "yes":
        mask = img_gray > 0
        img_gray[mask] = 255

        output_filename = path + "gt_processed_" + filename
        cv2.imwrite(output_filename, img_gray)
        print(f"Imagem processada salva em: {output_filename}")
    else:
        img_gray = remap_color_gt(img_gray)
        print("Níveis de cinza convertidas: ", np.unique(img_gray))

        output_filename = path + "gt_processed_" + filename
        cv2.imwrite(output_filename, img_gray)
        print(f"Imagem processada salva em: {output_filename}")


def remap_color_gt(img):
    label_cmap = [
        [0.29411764705882354, 0.4392156862745098, 0.7333333333333333],  # Cor 0
        [0.5882352941176471, 0.7607843137254902, 0.8666666666666667],  # Cor 1
        [0.8901960784313725, 0.9647058823529412, 0.9764705882352941],  # Cor 2
        [0.9803921568627451, 0.8745098039215686, 0.4666666666666667],  # Cor 3
        [0.9607843137254902, 0.47058823529411764, 0.29411764705882354],  # Cor 4
        [0.8470588235294118, 0.1568627450980392, 0.1411764705882353],  # Cor 5
    ]

    # Criar colormap atualizado
    label_cmap = ListedColormap(label_cmap)

    # Normalizar os valores da imagem para o intervalo [0, 1]
    img_normalized = img.astype(np.float32) / 5.0  # Normaliza para o intervalo [0, 1]

    # Mapeamento para as cores
    gt_color_array = label_cmap(
        img_normalized
    )  # Mapeia os valores normalizados para as cores do colormap

    # Converte para formato de imagem RGB e ajusta para o intervalo [0, 255]
    gt_color_img = (gt_color_array[:, :, :3] * 255).astype(
        np.uint8
    )  # Retorna para o intervalo [0, 255] e converte para uint8

    # Converter para BGR antes de salvar com OpenCV
    gt_color_img_bgr = cv2.cvtColor(gt_color_img, cv2.COLOR_RGB2BGR)

    return gt_color_img_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processa uma imagem, converte para escala de cinza e aplica uma máscara."
    )
    parser.add_argument("path", type=str, help="Caminho onde a imagem está localizada.")
    parser.add_argument("filename", type=str, help="Nome do arquivo da imagem.")
    parser.add_argument(
        "only_iou",
        type=str,
        help="Quer processar somente o nivel de pixel do iou ou toda a label.",
    )

    args = parser.parse_args()

    paint_mask(args.path, args.filename, args.only_iou)
