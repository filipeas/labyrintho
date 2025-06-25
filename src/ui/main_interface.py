import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QGraphicsView, QGraphicsScene, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen, QImage, QPixmap
from src.ui.mask_selection_window import MaskSelectionWindow
from src.controllers.image_loader import ImageLoader
from src.controllers.point_manager import PointManager
from src.controllers.segmenter import Segmenter

class MainInterface(QMainWindow):
    def __init__(self, config, model, device):
        super().__init__()
        self.setWindowTitle("Labyrintho - Seismic Facies Segmentation Tool")
        self.setGeometry(100, 100, 1000, 600)

        self.device = device

        self.image_loader = ImageLoader()
        self.point_manager = PointManager()
        self.segmenter = Segmenter(config["multimask_output"], config["height"], config["width"], model, device)

        self.point_items = []
        self.mask_windows = [] # manter referência viva
        
        self.scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.scene)

        self.status_bar = self.statusBar()
        self.update_status_bar()

        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.clicked.connect(self.load_image)

        self.save_button = QPushButton("Salvar Segmentação")
        self.save_button.clicked.connect(self.save_segmentation)

        self.reset_button = QPushButton("Reiniciar Segmentação")
        self.reset_button.clicked.connect(self.reset_segmentation)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.reset_button)
        layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.graphics_view)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.graphics_view.mousePressEvent = self.handle_mouse_click

        self.image = None
    
    def load_image(self):
        pixmap = self.image_loader.load_image()
        if pixmap:
            self.image = pixmap
            self.scene.clear() # clean scene
            self.scene.addPixmap(pixmap)
            self.point_manager.clear() # clean points
            self.segmenter.clear_prev_low_res_logits() # clean previous logits
            self.update_status_bar()
    
    def save_segmentation(self):
        if self.image:
            self.segmenter.segment(self.image, self.point_manager.points)
            # Aqui salvaria o resultado de fato
    
    def reset_segmentation(self):
        if self.image is None:
            return

        # Restaura a imagem original
        self.scene.clear()
        self.scene.addPixmap(self.image)

        # Limpa os pontos
        self.point_manager.clear()
        self.point_items.clear()

        # Limpa logits prévios do segmentador
        self.segmenter.clear_prev_low_res_logits()

        # Fecha janelas de máscaras se estiverem abertas
        for w in self.mask_windows:
            w.close()
        self.mask_windows.clear()

        self.update_status_bar()
    
    def predict(self):
        if self.image is None or len(self.point_manager.points) == 0 or self.image_loader.last_image_array is None:
            self.show_error("Não foi possível carregar a imagem ou os pontos. Analise o terminal para verificar o log corretamente.")
            print(20*"-")
            print("Imagem na cena foi carregado? ", self.image is not None)
            print("Pontos adicionados foram carregados? ", len(self.point_manager.points) > 0)
            print("Imagem em np.ndarray foi carregado? ", self.image_loader.last_image_array is not None)
            print(20*"-")

            # only for presentation: convert to uint8 RGB
            image_vis = self.image_loader.last_image_array
            if image_vis.dtype != np.uint8:
                image_vis = image_vis.astype(np.float32)
                image_vis -= image_vis.min()
                image_vis /= image_vis.max()
                image_vis *= 255
                image_vis = image_vis.astype(np.uint8)

            if image_vis.ndim == 2:
                image_vis = np.stack([image_vis] * 3, axis=-1)

            h, w, ch = image_vis.shape
            bytes_per_line = ch * w
            qimage = QImage(image_vis.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.scene.clear() # clean scene
            self.scene.addPixmap(pixmap)
            return
        low_res_logits, masks_logits = self.segmenter.segment(
            image=self.image_loader.last_image_array, points=self.point_manager.points
        )

        masks_logits = masks_logits.detach().cpu()  # shape [1, 3, H, W]
        base_image = self.image_loader.last_image_array.copy()
        self.best_idx = None

        def on_select(idx, threshold: float):
            self.best_idx = idx
            self.segmenter.update_prev_low_res_logits(
                low_res_logits[0, idx].unsqueeze(0).unsqueeze(0).to(self.device)
            )

            # Fecha janelas anteriores, se houver
            for w in self.mask_windows:
                w.close()
            self.mask_windows.clear()

            # Exibe máscara escolhida sobre a imagem principal
            selected_mask = torch.sigmoid(masks_logits[0, idx]).numpy()
            binary_mask = (selected_mask > threshold).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (self.image.width(), self.image.height()), interpolation=cv2.INTER_NEAREST)

            # Copia a imagem original
            base_image = self.image_loader.last_image_array.copy()
            base_image = self.image_loader.process_image_to_scene(image_np=base_image)

            # Cria uma cópia para o overlay (com vermelho nas regiões da máscara)
            overlay = base_image.copy()
            overlay[binary_mask == 1] = [255, 0, 0]

            # Aplica transparência apenas nas regiões da máscara
            alpha = 0.5
            combined = base_image.copy()
            mask_indices = binary_mask == 1
            combined[mask_indices] = (
                alpha * overlay[mask_indices] + (1 - alpha) * base_image[mask_indices]
            ).astype(np.uint8)

            h, w, c = combined.shape
            qimage = QImage(combined.data, w, h, combined.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            self.scene.clear()
            self.scene.addPixmap(pixmap)

            # Redesenha os pontos
            for x, y, label in self.point_manager.points:
                color = QColor("green") if label == 1 else QColor("red")
                self.draw_point(x, y, color)

        # Fecha janelas anteriores, se houver
        for w in self.mask_windows:
            w.close()
        self.mask_windows.clear()

        screen = self.geometry()  # Pega posição da janela principal
        start_x = screen.x() + 50
        start_y = screen.y() + 50
        offset = 30  # distância entre janelas

        # Abre janelas com as hipóteses (1 ou 3, depende do multimask_output)
        for i in range(masks_logits.shape[1]):
            window = MaskSelectionWindow(
                index=i, 
                base_image_np=base_image, 
                mask_logit_tensor=masks_logits[0, i], 
                callback_on_select=on_select
            )
            window.move(start_x + i * (600 + offset), start_y)  # lado a lado
            window.show()
            self.mask_windows.append(window)

    def handle_mouse_click(self, event):
        if self.image is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())
        if event.button() == Qt.LeftButton:
            self.point_manager.add_point(x, y, 1)
            self.draw_point(x, y, QColor("green"))
        elif event.button() == Qt.RightButton:
            self.point_manager.add_point(x, y, 0)
            self.draw_point(x, y, QColor("red"))
        self.update_status_bar()
        self.predict()
    
    def wheelEvent(self, event):
        super().wheelEvent(event)
    
    def update_status_bar(self):
        pos, neg = self.point_manager.count_points()
        self.status_bar.showMessage(
            f"Device: {self.device} | Positive points: {pos} | Negative points: {neg}"
        )
    
    def draw_point(self, x, y, color):
        radius = 5
        ellipse = self.scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, QPen(color), color)
        ellipse.setZValue(1)
        self.point_items.append(ellipse)
    
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo_last_point()
    
    def undo_last_point(self):
        removed = self.point_manager.undo_last_point()
        if removed is None:
            return

        if self.point_items:
            item = self.point_items.pop()
            self.scene.removeItem(item)
        self.update_status_bar()
        self.predict()
    
    def show_error(self, message):
        print("Erro:", message)
        QMessageBox.warning(None, "Erro", message)