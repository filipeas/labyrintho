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

class MainInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labyrintho - Seismic Facies Segmentation Tool")
        self.setGeometry(100, 100, 1000, 600)
        
        # Variável para armazenar o tipo de prompt
        self.current_prompt = "Positivo"
        self.points_positive = []  # Lista de pontos positivos (verde)
        self.points_negative = []  # Lista de pontos negativos (vermelho)
        
        # Variáveis de imagem
        self.pixmap_item = None  # Para exibir a imagem carregada
        self.gt_iou = None  # Imagem ddo ground truth para calcular iou
        
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

        self.graph_button = QPushButton("Selecionar Gráficos")
        self.graph_button.clicked.connect(self.open_graph_selector)
        
        # ComboBox para selecionar prompt
        self.prompt_selector = QComboBox()
        self.prompt_selector.addItems(["Selecione", "Pontos Positivos", "Pontos Negativos", "Borracha"])
        self.prompt_selector.currentIndexChanged.connect(self.set_prompt_type)

        # Seletor de gráficos
        self.graph_selector = QComboBox()
        self.graph_selector.addItems(["Selecione", "IoU vs. Pontos", "Outro Gráfico"])
        self.graph_selector.setVisible(False)
        
        # Layouts
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.load_annotations_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.graph_button)
        button_layout.addWidget(self.graph_selector)
        button_layout.addWidget(self.prompt_selector)
        button_layout.addStretch()
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.graphics_view)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Adicionar captura de clique do mouse
        self.graphics_view.mousePressEvent = self.mouse_press_event
    
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
            self.pixel_values = []  # Lista para armazenar valores de pixel únicos

        # Verificar se a máscara atual é diferente da anterior
        if not np.array_equal(mask, self.previous_mask):
            # Somar as máscaras
            self.gt_iou = np.clip(self.previous_mask + mask, 0, 1)
            self.previous_mask = self.gt_iou  # Atualizar a máscara anterior

            # Armazenar o valor de pixel único da máscara
            if pixel_value not in self.pixel_values:
                self.pixel_values.append(pixel_value)
        else:
            self.gt_iou = self.previous_mask  # Mantém a máscara atual

        print("self.gt_iou shape: ", self.gt_iou.shape)
        print("self.gt_iou unique: ", np.unique(self.gt_iou))

        # Mostrar a máscara atualizada
        plt.imshow(self.gt_iou, cmap="gray")
        plt.title("Máscara combinada do GT")
        plt.show()

        img = self.remap_color_gt(remap=self.pixel_values)
        
        # Converter a imagem colorida para QImage
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)

        # Atualizar o item de pixmap para o GT
        pixmap = QPixmap.fromImage(qimg)
        if hasattr(self, 'gt_pixmap_item') and self.gt_pixmap_item:
            self.scene.removeItem(self.gt_pixmap_item)  # Remove o antigo item GT
        self.gt_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.gt_pixmap_item.setOffset(0, self.pixmap_item.pixmap().height())  # Ajusta a posição abaixo da imagem original
        self.scene.addItem(self.gt_pixmap_item)

        # Atualizar a visualização para manter o aspecto
        self.graphics_view.setScene(self.scene)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_prompt_type(self):
        """Atualiza o tipo de prompt com base na seleção do ComboBox."""
        self.current_prompt = self.prompt_selector.currentText()
    
    def load_image(self):
        """Carregar uma imagem e exibi-la na área de visualização."""
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
                self.image_array_gt = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Carrega em escala de cinza
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
        folder_path = QFileDialog.getExistingDirectory(self, "Selecionar Pasta com Arquivos")
        if folder_path:
            try:
                # Carregar imagem original
                original_image_path = os.path.join(folder_path, "imagem_original.tiff")
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
                    self.image_array_gt = cv2.imread(original_image_path_gt, cv2.IMREAD_GRAYSCALE)  # Carrega em escala de cinza
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
            except Exception as e:
                print(f"Erro ao carregar arquivos: {e}")
    
    def save_image_and_points(self):
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
        graph_name = self.graph_selector.currentText()
        if graph_name != "Selecione":  # Ignorar quando a opção "Selecione" está ativa
            if self.graph_window:
                self.graph_window.close()  # Fechar a janela anterior se existir
            self.graph_window = GraphViewer(graph_name)  # Manter a referência
            self.graph_window.show()
    
    def mouse_press_event(self, event):
        """Captura o clique do mouse para adicionar pontos na imagem."""
        if self.pixmap_item:
            scene_pos = self.graphics_view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            
            if self.current_prompt == "Pontos Positivos":
                self.points_positive.append((x, y))  # Adiciona ponto positivo
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

        # Remover o valor de pixel da lista de valores de pixel
        if pixel_value in self.pixel_values:
            self.pixel_values.remove(pixel_value)
        
        print("self.gt_iou shape: ", self.gt_iou.shape)
        print("self.gt_iou unique: ", np.unique(self.gt_iou))

        # Mostrar a máscara atualizada
        plt.imshow(self.gt_iou, cmap="gray")
        plt.title("Máscara combinada do GT")
        plt.show()
        
        # Atualizar a imagem GT
        img = self.remap_color_gt(remap=self.pixel_values)

        # Converter a imagem colorida para QImage
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)

        # Atualizar o item de pixmap para o GT
        pixmap = QPixmap.fromImage(qimg)
        if hasattr(self, 'gt_pixmap_item') and self.gt_pixmap_item:
            self.scene.removeItem(self.gt_pixmap_item)  # Remove o antigo item GT
        self.gt_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.gt_pixmap_item.setOffset(0, self.pixmap_item.pixmap().height())  # Ajusta a posição abaixo da imagem original
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
                    print(f"Ponto removido: {point}")
                    self.update_image_with_points()
                    self.erase_regions_event(x, y) # apagando regiao
                    return  # Remove apenas um ponto
        
        # Remove o ponto positivo mais próximo, se existir
        for point_list in [self.points_negative]:
            for point in point_list:
                if abs(point[0] - x) <= tolerance and abs(point[1] - y) <= tolerance:
                    point_list.remove(point)
                    print(f"Ponto removido: {point}")
                    self.update_image_with_points()
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
    
    def plot_iou_graph(self):
        points = np.arange(1, 11)
        iou = np.log(points) / np.log(10)
        
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(points, iou, marker='o', label="IoU")
        ax.set_title("IoU vs. Pontos")
        ax.set_xlabel("Pontos Adicionados")
        ax.set_ylabel("IoU")
        ax.legend()
        self.canvas.draw()
    
    def plot_other_graph(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(x, y, marker='', label="Seno")
        ax.set_title("Outro Gráfico")
        ax.set_xlabel("Eixo X")
        ax.set_ylabel("Eixo Y")
        ax.legend()
        self.canvas.draw()