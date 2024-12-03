import os
import numpy as np
import cv2
import tifffile as tiff
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QFileDialog, 
    QVBoxLayout, QHBoxLayout, QWidget, QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem, QComboBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt5.QtCore import Qt, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
from torchmetrics import JaccardIndex

class MainInterface(QMainWindow):
    def __init__(self, model, device):
        super().__init__()
        self.setWindowTitle("Labyrintho - Seismic Facies Segmentation Tool")
        self.model = model
        self.device = device

        print("Camadas do modelo: ", self.model)
        self.setGeometry(100, 100, 1000, 600)
        
        # Variável para armazenar o tipo de prompt
        self.current_prompt = "Positivo"
        self.points_positive = []  # Lista de pontos positivos (verde)
        self.points_negative = []  # Lista de pontos negativos (vermelho)
        
        # Variáveis de imagem
        self.pixmap_item = None  # Para exibir a imagem carregada
        self.gt_iou = None  # Imagem ddo ground truth para calcular iou
        self.current_graph = None # controla o grafico atual
        self.graph_window = None # janela do grafico
        
        # Área de exibição da imagem
        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        
        # Botões laterais
        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.clicked.connect(self.load_image)

        self.load_annotations_button = QPushButton("Carregar Marcações")
        self.load_annotations_button.clicked.connect(self.load_annotations)
        
        self.save_button = QPushButton("Salvar Imagem e Pontos")
        self.save_button.clicked.connect(self.save_image_and_points)

        # self.graph_button = QPushButton("Selecionar Gráficos")
        # self.graph_button.clicked.connect(self.open_graph_selector)

        # Botão para realizar segmentação
        self.segment_button = QPushButton("Realizar Segmentação")
        self.segment_button.setStyleSheet("background-color: green; color: white;")
        self.segment_button.setVisible(False)  # Inicialmente oculto
        self.segment_button.clicked.connect(self.perform_segmentation)
        
        # ComboBox para selecionar prompt
        self.prompt_selector = QComboBox()
        self.prompt_selector.addItems(["Selecione o Prompt", "Pontos Positivos", "Pontos Negativos", "Borracha"])
        self.prompt_selector.currentIndexChanged.connect(self.set_prompt_type)

        # Seletor de gráficos
        self.graph_selector = QComboBox()
        self.graph_selector.addItems(["Selecione o Gráfico", "IoU vs. Pontos", "Outro Gráfico"])
        self.graph_selector.currentIndexChanged.connect(self.show_graph)
        
        # Layouts
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.load_annotations_button)
        button_layout.addWidget(self.save_button)
        # button_layout.addWidget(self.graph_button)
        button_layout.addWidget(self.graph_selector)
        button_layout.addWidget(self.prompt_selector)
        button_layout.addWidget(self.segment_button)
        button_layout.addStretch()
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.graphics_view)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Adicionar captura de clique do mouse
        self.graphics_view.mousePressEvent = self.mouse_press_event
    
    def update_button_visibility(self):
        """Atualiza a visibilidade do botão de segmentação."""
        if self.points_positive and len(self.points_positive) > 0 and hasattr(self, "gt_iou"):
            self.segment_button.setVisible(True)
        else:
            self.segment_button.setVisible(False)
    
    def perform_segmentation(self):
        """Executa a segmentação utilizando o modelo e salva métricas no log."""
        if not self.points_positive:
            print("Nenhum ponto positivo disponível.")
            return
        
        if self.gt_iou is None:
            print("Ground truth não carregado.")
            return
        
        # preparando pontos e suas labels
        point_coords = np.array(self.points_positive + self.points_negative)
        point_labels = np.array([1] * len(self.points_positive) + [0] * len(self.points_negative))
        # print("Pontos:", point_coords.shape)
        # print("Labels:", point_labels.shape)

        # no caso do sam, é passado a classe Predictor e não o modelo...
        if self.image_array.dtype != np.uint8:
            tiff_image = ((self.image_array - self.image_array.min()) / (self.image_array.max() - self.image_array.min()) * 255).astype(np.uint8)
        else:
            tiff_image = self.image_array
        _, png_img = cv2.imencode('.png', tiff_image)
        decoded_image = cv2.imdecode(np.frombuffer(png_img, np.uint8), cv2.IMREAD_UNCHANGED)
        print(type(decoded_image), decoded_image.shape)

        self.model.set_image(decoded_image)

        # plotando
        # plt.clf()
        # plt.figure(figsize=(10, 8))
        # plt.imshow(decoded_image, cmap='gray')
        # for (x, y), label in zip(point_coords, point_labels):
        #     color = 'green' if label == 1 else 'red'
        #     plt.scatter(x, y, color=color, label='Positive' if label == 1 else 'Negative')

        # # Configurar legendas
        # handles = [
        #     plt.Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green'),
        #     plt.Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red')
        # ]
        # plt.legend(handles=handles)
        # plt.axis('off')
        # plt.show()

        # Preparar dados para o modelo
        masks, scores, logits = self.model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False, # controla ambiguidade
        )

        # Criar subplots para exibir a imagem com todas as máscaras
        num_masks = masks.shape[0]
        plt.clf()
        fig, axes = plt.subplots(1, num_masks, figsize=(5 * num_masks, 5))

        # Caso haja apenas uma máscara, torne 'axes' um array de um único elemento
        if num_masks == 1:
            axes = [axes]

        # Plotar a imagem original e adicionar a máscara e os pontos sobre cada uma
        for i, (mask, score, ax) in enumerate(zip(masks, scores, axes)):
            ax.imshow(decoded_image, cmap='gray')  # Exibe a imagem original
            ax.imshow(mask, cmap='jet', alpha=0.5)  # Exibe a máscara sobre a imagem
            for (x, y), label in zip(point_coords, point_labels):
                color = 'green' if label == 1 else 'red'
                ax.scatter(x, y, color=color, label='Positive' if label == 1 else 'Negative')

            # Título com o score da máscara
            ax.set_title(f"Mask {i+1}\nScore: {score:.3f}", fontsize=12)
            ax.axis('off')

        # Ajuste para evitar sobreposição
        plt.tight_layout()
        plt.show()
        
        # Calcular IoU
        # Converta a máscara de predição e o ground truth para tensores PyTorch
        gt_tensor = torch.tensor(self.gt_iou).to(self.device)  # Converta para tensor 2D e mova para GPU
        pred_tensor = torch.tensor(masks.squeeze()).to(self.device)  # Remover a dimensão extra e mover para GPU
        # print("shape gt_tensor: ", gt_tensor.shape)
        # print("shape pred_tensor: ", pred_tensor.shape)
        # Definir o número de classes (2 classes: fundo e objeto)
        num_classes = 2

        # Inicializar o JaccardIndex (IoU)
        miou_metric = JaccardIndex(task="multiclass", num_classes=num_classes).to(self.device)
        iou_score = miou_metric(pred_tensor, gt_tensor)

        self.save_image_and_points(save_segmentation=True, mask=pred_tensor, iou_score=iou_score.item())

    def update_segmentation(self, x, y):
        """Atualiza a segmentação baseada no ponto positivo."""
        if self.image_array_gt is None:
            print("Ground truth não carregado.")
            return

        # Obter o nível de cinza do ground truth no ponto selecionado
        pixel_value = self.image_array_gt[y, x]

        # Criar uma máscara para todos os pixels com o mesmo nível de cinza
        mask = (self.image_array_gt == pixel_value).astype(np.uint8)

        # Inicializar self.previous_mask se ainda não existir
        if not hasattr(self, 'previous_mask') or self.previous_mask is None:
            self.previous_mask = np.zeros_like(self.image_array_gt, dtype=np.uint8)
        
        if not hasattr(self, 'pixel_values'):
            self.pixel_values = {}  # Lista para armazenar valores de pixel únicos

        # Verificar se a máscara atual é diferente da anterior
        if not np.array_equal(mask, self.previous_mask):
            # Somar as máscaras
            self.gt_iou = np.clip(self.previous_mask + mask, 0, 1)
            self.previous_mask = self.gt_iou  # Atualizar a máscara anterior
        else:
            self.gt_iou = self.previous_mask  # Mantém a máscara atual
        
        # Armazenar o valor de pixel único da máscara
        if pixel_value not in self.pixel_values:
            self.pixel_values[pixel_value] = []
        self.pixel_values[pixel_value].append((x,y))

        # Mostrar a máscara atualizada
        plt.clf()
        plt.imshow(self.gt_iou, cmap="gray")
        plt.title("Máscara combinada do GT")
        plt.show()

        img = self.remap_color_gt(remap=list(self.pixel_values.keys()))

        # Converter a imagem colorida para QImage
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)

        # Atualizar o item de pixmap para o GT
        pixmap = QPixmap.fromImage(qimg)
        if hasattr(self, 'gt_pixmap_item') and self.gt_pixmap_item:
            self.scene.removeItem(self.gt_pixmap_item)  # Remove o antigo item GT
        self.gt_pixmap_item = QGraphicsPixmapItem(pixmap)
        
        # Verificar a orientação da imagem original
        img_width = self.pixmap_item.pixmap().width()
        img_height = self.pixmap_item.pixmap().height()

        if img_height > img_width:  # Imagem maior na vertical
            # Posicionar o GT à direita da imagem
            self.gt_pixmap_item.setOffset(img_width, 0)
        else:  # Imagem maior na horizontal
            # Posicionar o GT abaixo da imagem
            self.gt_pixmap_item.setOffset(0, img_height)

        self.scene.addItem(self.gt_pixmap_item)

        # Atualizar a visualização para manter o aspecto
        self.graphics_view.setScene(self.scene)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_prompt_type(self):
        """Atualiza o tipo de prompt com base na seleção do ComboBox."""
        self.current_prompt = self.prompt_selector.currentText()
    
    def load_image(self):
        """Carregar uma imagem e exibi-la na área de visualização."""

        # resetando variaveis
        self.points_positive = []
        self.points_negative = []
        self.pixmap_item = None
        self.gt_iou = None
        self.current_graph = None
        self.graph_window = None
        self.previous_mask = None
        self.pixel_values = {}

        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Imagens (*.tiff *.tif *.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                self.image_array = tiff.imread(file_path)  # Carrega a imagem como um array numpy
                img = self.image_array
                print(f"Imagem carregada com shape: {img.shape}, dtype: {img.dtype}")
                # Salvar o caminho do arquivo original e sua extensão
                self.original_file_path = file_path
                self.original_file_extension = os.path.splitext(file_path)[1]

                # Normaliza a imagem se necessário
                if img.dtype != np.uint8:
                    img = (img / img.max() * 255).astype(np.uint8)

                # Garantir que tenha 3 canais (RGB)
                if len(img.shape) == 2:  # Imagem em escala de cinza
                    img = np.stack((img,) * 3, axis=-1)

                # Converte o numpy array para QImage
                qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)

                # Converter QImage para QPixmap
                pixmap = QPixmap.fromImage(qim)

                self.scene.clear()
                self.pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.pixmap_item)
                self.graphics_view.setScene(self.scene)
                self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                self.setWindowTitle(f"Imagem: {file_path}")

                # Salvar o pixmap original como buffer
                self.original_pixmap = pixmap.copy()

                # Limpar os pontos ao carregar uma nova imagem
                self.points_positive.clear()
                self.points_negative.clear()

                self.load_ground_truth()
            except Exception as e:
                print(f"Erro ao carregar a imagem: {e}")
    
    def remap_color_gt(self, remap=None):
        label_cmap = [
            [0.29411764705882354, 0.4392156862745098, 0.7333333333333333],  # Cor 0
            [0.5882352941176471, 0.7607843137254902, 0.8666666666666667],  # Cor 1
            [0.8901960784313725, 0.9647058823529412, 0.9764705882352941],  # Cor 2
            [0.9803921568627451, 0.8745098039215686, 0.4666666666666667],  # Cor 3
            [0.9607843137254902, 0.47058823529411764, 0.29411764705882354],  # Cor 4
            [0.8470588235294118, 0.1568627450980392, 0.1411764705882353],  # Cor 5
        ]

        # Aplicar remapeamento, se fornecido
        if remap:
            for idx in remap:
                if 0 <= idx < len(label_cmap):  # Garantir que o índice está dentro do intervalo válido
                    label_cmap[idx] = [1, 1, 1]
        
        # Criar colormap atualizado
        label_cmap = ListedColormap(label_cmap)

        # Criar uma imagem colorida a partir dos valores mapeados
        gt_color_array = label_cmap(self.image_array_gt / 5)  # Normaliza para o intervalo [0, 1]
        gt_color_img = (gt_color_array[:, :, :3] * 255).astype(np.uint8)  # Converte para formato de imagem RGB
        
        return gt_color_img
    
    def load_ground_truth(self):
        """Carregar e exibir o ground truth (verdadeiro) da imagem."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Ground Truth", "", "Imagens (*.tiff *.tif *.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            try:
                # Carregar a imagem usando OpenCV
                self.image_array_gt = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                print("Níveis de cinza detectados: ", np.unique(self.image_array_gt))
                
                # Salvar o caminho do arquivo original e sua extensão
                self.original_file_path_gt = file_path
                self.original_file_extension_gt = os.path.splitext(file_path)[1]

                # Verificar se a imagem foi carregada corretamente
                if self.image_array_gt is None:
                    raise ValueError("Não foi possível carregar a imagem.")

                gt_color_img = self.remap_color_gt()

                # Converter a imagem colorida para QImage
                gt_img_qimage = QImage(gt_color_img.data, gt_color_img.shape[1], gt_color_img.shape[0], gt_color_img.strides[0], QImage.Format_RGB888)

                # Converter QImage para QPixmap
                gt_pixmap = QPixmap.fromImage(gt_img_qimage)

                # Verificar a orientação da imagem
                img_width = self.pixmap_item.pixmap().width()
                img_height = self.pixmap_item.pixmap().height()

                # Adicionar a barra de separação
                separation_bar_height = 10  # Defina a altura da barra de separação
                separation_bar_color = QColor(200, 200, 200)  # Cor da barra (cinza claro)

                if img_height > img_width:  # Imagem maior na vertical
                    # Colocar o GT à direita da imagem
                    self.gt_pixmap_item = QGraphicsPixmapItem(gt_pixmap)
                    self.gt_pixmap_item.setOffset(img_width, 0)  # Alinha o GT à direita
                else:  # Imagem maior na horizontal
                    # Colocar o GT abaixo da imagem
                    self.gt_pixmap_item = QGraphicsPixmapItem(gt_pixmap)
                    self.gt_pixmap_item.setOffset(0, img_height)  # Coloca o GT abaixo

                self.scene.addItem(self.gt_pixmap_item)
                self.graphics_view.setScene(self.scene)
                self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

                # Salvar o pixmap do ground truth como buffer
                self.ground_truth_pixmap = gt_pixmap.copy()

            except Exception as e:
                print(f"Erro ao carregar o ground truth: {e}")
    
    def load_annotations(self):
        """Carrega a imagem original e as marcações de uma pasta."""

        # resetando variaveis
        self.points_positive = []
        self.points_negative = []
        self.pixmap_item = None
        self.gt_iou = None
        self.current_graph = None
        self.graph_window = None
        self.previous_mask = None
        self.pixel_values = {}
        
        folder_path = QFileDialog.getExistingDirectory(self, "Selecionar Pasta com Arquivos")
        if folder_path:
            try:
                # Verificar se a imagem original existe
                possible_extensions = ["imagem_original.tiff", "imagem_original.tif"]
                original_image_path = None
                
                for ext in possible_extensions:
                    path = os.path.join(folder_path, ext)
                    if os.path.exists(path):
                        original_image_path = path
                        break
                
                if original_image_path:
                    self.image_array = tiff.imread(original_image_path)
                else:
                    print("Nenhuma imagem original encontrada com as extensões .tiff ou .tif.")
                if os.path.exists(original_image_path):
                    self.image_array = tiff.imread(original_image_path)
                    img = self.image_array
                    print(f"Imagem original carregada: {img.shape}")
                    # Salvar o caminho do arquivo original e sua extensão
                    self.original_file_path = original_image_path
                    self.original_file_extension = os.path.splitext(original_image_path)[1]

                    if img.dtype != np.uint8:
                        img = (img / img.max() * 255).astype(np.uint8)
                    qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qim)

                    self.scene.clear()
                    self.pixmap_item = QGraphicsPixmapItem(pixmap)
                    self.scene.addItem(self.pixmap_item)
                    self.graphics_view.setScene(self.scene)
                    self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

                    # Salvar o pixmap original como buffer
                    self.original_pixmap = pixmap.copy()

                # Carregar GT
                original_image_path_gt = os.path.join(folder_path, "gt.png")
                if os.path.exists(original_image_path_gt):
                    self.image_array_gt = cv2.imread(original_image_path_gt, cv2.IMREAD_UNCHANGED)
                    print("Níveis de cinza detectados: ", np.unique(self.image_array_gt))
                    # Salvar o caminho do arquivo original e sua extensão
                    self.original_file_path_gt = original_image_path_gt
                    self.original_file_extension_gt = os.path.splitext(original_image_path_gt)[1]

                    # Verificar se a imagem foi carregada corretamente
                    if self.image_array_gt is None:
                        raise ValueError("Não foi possível carregar a imagem.")
                    
                    gt_color_img = self.remap_color_gt()

                    # Converter a imagem colorida para QImage
                    gt_img_qimage = QImage(gt_color_img.data, gt_color_img.shape[1], gt_color_img.shape[0], gt_color_img.strides[0], QImage.Format_RGB888)
                    # print("gt_img_qimage:", gt_img_qimage)

                    # Converter QImage para QPixmap
                    gt_pixmap = QPixmap.fromImage(gt_img_qimage)

                    # Verificar a orientação da imagem
                    img_width = self.pixmap_item.pixmap().width()
                    img_height = self.pixmap_item.pixmap().height()

                    # Adicionar a barra de separação
                    separation_bar_height = 10  # Defina a altura da barra de separação
                    separation_bar_color = QColor(200, 200, 200)  # Cor da barra (cinza claro)

                    if img_height > img_width:  # Imagem maior na vertical
                        # Colocar o GT à direita da imagem
                        self.gt_pixmap_item = QGraphicsPixmapItem(gt_pixmap)
                        self.gt_pixmap_item.setOffset(img_width, 0)  # Alinha o GT à direita
                    else:  # Imagem maior na horizontal
                        # Colocar o GT abaixo da imagem
                        self.gt_pixmap_item = QGraphicsPixmapItem(gt_pixmap)
                        self.gt_pixmap_item.setOffset(0, img_height)  # Coloca o GT abaixo

                    self.scene.addItem(self.gt_pixmap_item)
                    self.graphics_view.setScene(self.scene)
                    self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

                    # Salvar o pixmap do ground truth como buffer
                    self.ground_truth_pixmap = gt_pixmap.copy()

                # Carregar marcações
                json_file_path = os.path.join(folder_path, "marcacoes.json")
                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as json_file:
                        points_data = json.load(json_file)
                        self.points_positive = points_data.get("pontos_positivos", [])
                        self.points_negative = points_data.get("pontos_negativos", [])
                    
                    self.update_image_with_points()

                    print("Marcações carregadas com sucesso.")
                else:
                    print("Arquivo JSON não encontrado.")
                
                # Atualiza GT IOU
                for point_list in [self.points_positive]:
                    for point in point_list:
                        self.update_segmentation(point[0], point[1])
                
                # carrega botão de segmentar
                self.update_button_visibility()
            except Exception as e:
                print(f"Erro ao carregar arquivos: {e}")
    
    def save_image_and_points(self, save_segmentation=False, mask=None, iou_score=None):
        """Salva a imagem original, a imagem com os pontos e o arquivo JSON em uma pasta específica."""
        base_folder_path = QFileDialog.getExistingDirectory(self, "Selecionar Local para Salvar")
        if base_folder_path:
            # Pedir ao usuário o nome da pasta
            folder_name, ok = QInputDialog.getText(self, "Nome da Pasta", "Digite o nome da pasta:")
            if ok and folder_name.strip():
                # Criar a subpasta com o nome fornecido pelo usuário
                save_folder = os.path.join(base_folder_path, folder_name.strip())
                os.makedirs(save_folder, exist_ok=True)

                # Salvar imagem original
                original_image_path = os.path.join(save_folder, f"imagem_original{self.original_file_extension}")
                tiff.imwrite(original_image_path, self.image_array)  # Salva o array NumPy como TIFF

                # Salvar GT
                original_image_gt_path = os.path.join(save_folder, f"gt{self.original_file_extension_gt}")
                tiff.imwrite(original_image_gt_path, self.image_array_gt)  # Salva como png

                # Salvar GT para IOU
                original_image_gt_path = os.path.join(save_folder, f"gt_iou.png")
                tiff.imwrite(original_image_gt_path, self.gt_iou)  # Salva como png

                # Salvar imagem marcada
                marked_image_path = os.path.join(save_folder, "imagem_marcada.png")
                marked_pixmap = self.pixmap_item.pixmap()  # Obter a imagem com as marcações diretamente da cena

                # Salvar a imagem com as marcações
                marked_pixmap.save(marked_image_path)

                # Salvar pontos em JSON
                points_data = {
                    "pontos_positivos": self.points_positive,
                    "pontos_negativos": self.points_negative
                }
                json_file_path = os.path.join(save_folder, "marcacoes.json")
                with open(json_file_path, "w") as json_file:
                    json.dump(points_data, json_file, indent=4)
                
                if save_segmentation and mask is not None:
                    # salvar mascara
                    mask_iou_path = os.path.join(save_folder, f"mask_segmentada_{iou_score:.4f}_{len(self.points_positive)}_{len(self.points_negative)}.png")
                    cv2.imwrite(mask_iou_path, (mask * 255).cpu().numpy().astype(np.uint8))
                    # Salvar pontos em JSON
                    json_file_path = os.path.join(save_folder, "segmentacoes.json")
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as json_file:
                            existing_data = json.load(json_file)
                    else:
                        existing_data = []

                    result = {
                        "iou": iou_score,
                        "pontos_positivos": len(self.points_positive),
                        "pontos_negativos": len(self.points_negative),
                    }

                    existing_data.append(result)

                    with open(json_file_path, "w") as json_file:
                        json.dump(existing_data, json_file, indent=4)

                print(f"Arquivos salvos em: {save_folder}")
            else:
                print("Nenhum nome de pasta foi fornecido. Operação cancelada.")
    
    def open_graph_selector(self):
        """Exibir ou ocultar o seletor de gráficos."""
        self.graph_selector.setVisible(not self.graph_selector.isVisible())
        if self.graph_selector.isVisible():
            self.graph_selector.currentIndexChanged.connect(self.show_graph)
    
    def show_graph(self):
        """Abrir a tela para exibir o gráfico selecionado."""
        selected_graph = self.graph_selector.currentText()
        if selected_graph == "Selecione o Gráfico":
            return  # Não faz nada se a opção padrão for selecionada

        if self.graph_window:
            self.graph_window.close()  # Fecha a janela anterior, se existir

        self.graph_window = GraphViewer(selected_graph)
        self.graph_window.show()
    
    def mouse_press_event(self, event):
        """Captura o clique do mouse para adicionar pontos na imagem."""
        if self.pixmap_item:
            scene_pos = self.graphics_view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            
            if self.current_prompt == "Pontos Positivos":
                self.points_positive.append((x, y))  # Adiciona ponto positivo
                self.update_button_visibility()
                print(f"Ponto positivo adicionado em: ({x}, {y})")
                self.update_segmentation(x, y)
            elif self.current_prompt == "Pontos Negativos":
                self.points_negative.append((x, y))  # Adiciona ponto negativo
                print(f"Ponto negativo adicionado em: ({x}, {y})")
            elif self.current_prompt == "Borracha":
                self.erase_point(x, y)  # Remove o ponto mais próximo
            else:
                print(f"Tipo de prompt não suportado: {self.current_prompt}")
                
            self.update_image_with_points()
    
    def erase_regions_event(self, x, y):
        if self.image_array_gt is None:
            print("Ground truth não carregado.")
            return
        
        # Obter o nível de cinza do ground truth no ponto selecionado
        pixel_value = self.image_array_gt[y, x]

        # Criar uma máscara para todos os pixels com o mesmo nível de cinza
        mask_to_remove = (self.image_array_gt == pixel_value).astype(np.uint8)

        # Remover a região da máscara de gt_iou
        self.gt_iou = np.clip(self.gt_iou - mask_to_remove, 0, 1)
        self.previous_mask = self.gt_iou  # Atualizar a máscara anterior

        # Remover o valor de pixel da lista de valores de pixel
        if not self.pixel_values[pixel_value]:
            del self.pixel_values[pixel_value]

        # Mostrar a máscara atualizada
        plt.clf()
        plt.imshow(self.gt_iou, cmap="gray")
        plt.title("Máscara combinada do GT")
        plt.show()
        
        # Atualizar a imagem GT
        img = self.remap_color_gt(remap=list(self.pixel_values.keys()))

        # Converter a imagem colorida para QImage
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)

        # Atualizar o item de pixmap para o GT
        pixmap = QPixmap.fromImage(qimg)
        if hasattr(self, 'gt_pixmap_item') and self.gt_pixmap_item:
            self.scene.removeItem(self.gt_pixmap_item)  # Remove o antigo item GT
        self.gt_pixmap_item = QGraphicsPixmapItem(pixmap)
        
        # Verificar a orientação da imagem original
        img_width = self.pixmap_item.pixmap().width()
        img_height = self.pixmap_item.pixmap().height()

        if img_height > img_width:  # Imagem maior na vertical
            # Posicionar o GT à direita da imagem
            self.gt_pixmap_item.setOffset(img_width, 0)
        else:  # Imagem maior na horizontal
            # Posicionar o GT abaixo da imagem
            self.gt_pixmap_item.setOffset(0, img_height)
        
        self.scene.addItem(self.gt_pixmap_item)

        # Atualizar a visualização para manter o aspecto
        self.graphics_view.setScene(self.scene)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

        print(f"Região com valor de pixel {pixel_value} removida.")

    def erase_point(self, x, y):
        """Remove o ponto mais próximo das listas e da imagem."""
        # Define a tolerância para encontrar o ponto mais próximo
        tolerance = 10  # Pixels
        
        """ esse loop é separado pois existe a regra erase_regions_event() que só pode ser aplicado aos pontos positivos """
        # Remove o ponto positivo mais próximo, se existir
        for point_list in [self.points_positive]:
            for point in point_list:
                if abs(point[0] - x) <= tolerance and abs(point[1] - y) <= tolerance:
                    point_list.remove(point)
                    print(f"Ponto removido de points_positive: {point}")
                    
                    pixel_value = None
                    for key, value in self.pixel_values.items():
                        for save_point in value:
                            print(save_point, type(save_point))
                            if tuple(point) == save_point:
                                pixel_value = key
                                break
                    
                    print("Pixel value achado: ", pixel_value)
                    
                    if pixel_value is not None:
                        # achou a key (nivel de cinza)
                        self.pixel_values[pixel_value].remove(tuple(point))
                        print(f"Ponto removido do pixel_value ({pixel_value}): {point}")

                        if not self.pixel_values[pixel_value]:
                            self.erase_regions_event(point[0], point[1]) # apagando regiao
                            print(f"Região apagada para o ponto {point} no pixel_value {pixel_value}")
                    
                    # self.update_image_with_points() # Atualiza a imagem com os pontos modificados
                    return  # Remove apenas um ponto
        
        # Remove o ponto positivo mais próximo, se existir
        for point_list in [self.points_negative]:
            for point in point_list:
                if abs(point[0] - x) <= tolerance and abs(point[1] - y) <= tolerance:
                    point_list.remove(point)
                    print(f"Ponto removido de points_negative: {point}")
                    # self.update_image_with_points()
                    return  # Remove apenas um ponto
    
    def update_image_with_points(self):
        """Atualiza a imagem desenhando os pontos na imagem."""
        # Cria uma cópia do buffer original
        pixmap = self.original_pixmap.copy()

        # Configura o QPainter para desenhar sobre a cópia
        painter = QPainter(pixmap)

        # Desenha pontos positivos (verdes)
        painter.setPen(QColor(0, 255, 0))  # Cor verde
        painter.setBrush(QColor(0, 255, 0, 100))  # Cor verde para o preenchimento, com opacidade (alfa 100)
        for point in self.points_positive:
            painter.drawEllipse(QPoint(int(point[0]), int(point[1])), 5, 5)

        # Desenha pontos negativos (vermelhos)
        painter.setPen(QColor(255, 0, 0))  # Cor vermelha
        painter.setBrush(QColor(255, 0, 0, 100))  # Cor vermelha para o preenchimento, com opacidade (alfa 100)
        for point in self.points_negative:
            painter.drawEllipse(QPoint(int(point[0]), int(point[1])), 5, 5)

        painter.end()

        # Atualiza o item da imagem no QGraphicsScene
        self.pixmap_item.setPixmap(pixmap)

class GraphViewer(QMainWindow):
    def __init__(self, graph_name):
        super().__init__()
        self.setWindowTitle(f"Visualizando: {graph_name}")
        self.setGeometry(150, 150, 800, 600)
        
        self.canvas = FigureCanvas(plt.figure())
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Plotar o gráfico com base no nome
        if graph_name == "IoU vs. Pontos":
            self.plot_iou_graph()
        elif graph_name == "Outro Gráfico":
            self.plot_other_graph()
    
    def read_json(self):
        # Abre um diálogo para selecionar o arquivo JSON
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecione o arquivo JSON", "", "JSON Files (*.json)")
        if not file_path:
            return None

        with open(file_path, "r") as file:
            data = json.load(file)

        # Remove duplicatas com base em tuplas únicas de (iou, pontos_positivos, pontos_negativos)
        unique_data = list({(item["iou"], item["pontos_positivos"], item["pontos_negativos"]): item for item in data}.values())
        return unique_data
    
    def plot_iou_graph(self):
        data = self.read_json()
        if data is None:
            return

        # Processar os dados para gráficos
        pontos_positivos = []
        pontos_negativos = []
        iou_por_ponto_positivo = []
        iou_por_ponto_negativo = []

        for item in data:
            pontos_positivos.append(item["pontos_positivos"])
            pontos_negativos.append(item["pontos_negativos"])
            iou_por_ponto_positivo.append(item["iou"])
            iou_por_ponto_negativo.append(item["iou"])

        # Garantir dados únicos
        pontos_positivos = np.array(pontos_positivos)
        pontos_negativos = np.array(pontos_negativos)
        iou_por_ponto_positivo = np.array(iou_por_ponto_positivo)
        iou_por_ponto_negativo = np.array(iou_por_ponto_negativo)

        # Ordenar por pontos positivos
        sorted_indices_positivos = np.argsort(pontos_positivos)
        pontos_positivos = pontos_positivos[sorted_indices_positivos]
        iou_por_ponto_positivo = iou_por_ponto_positivo[sorted_indices_positivos]

        # Ordenar por pontos negativos
        sorted_indices_negativos = np.argsort(pontos_negativos)
        pontos_negativos = pontos_negativos[sorted_indices_negativos]
        iou_por_ponto_negativo = iou_por_ponto_negativo[sorted_indices_negativos]

        # Criar o gráfico
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        
        # Plotar IoU por pontos positivos
        ax.plot(pontos_positivos, iou_por_ponto_positivo, marker='o', label="IoU por Pontos Positivos", color='blue')
        
        # Plotar IoU por pontos negativos
        ax.plot(pontos_negativos, iou_por_ponto_negativo, marker='x', label="IoU por Pontos Negativos", color='red')
        
        # Personalização do gráfico
        ax.set_title("IoU vs. Pontos Positivos e Negativos")
        ax.set_xlabel("Pontos")
        ax.set_ylabel("IoU")
        ax.legend(loc='best')
        self.canvas.draw()
    
    def plot_other_graph(self):
        pass