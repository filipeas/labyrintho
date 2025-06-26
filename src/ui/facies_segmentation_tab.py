import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QGraphicsView,
    QGraphicsScene,
    QMessageBox,
    QPushButton,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor, QPen, QImage, QPixmap
from src.ui.mask_selection_window import MaskSelectionWindow
from src.controllers.image_loader import ImageLoader
from src.controllers.point_manager import PointManager
from src.controllers.segmenter import Segmenter
from src.ui.hypotheses_window import HypothesesWindow


class FaciesSegmentationTab(QWidget):
    def __init__(
        self,
        name: str,
        image: QPixmap,
        image_loader: ImageLoader,
        config,
        model,
        device,
    ):
        super().__init__()

        self.name = name
        self.device = device
        self.image = image.copy()
        self.image_loader = image_loader
        self.point_manager = PointManager()
        self.segmenter = Segmenter(
            config["multimask_output"], config["height"], config["width"], model, device
        )

        self.point_items = []
        self.mask_windows = []
        self.binary_mask = None
        self.best_idx = None
        self.edit_mode = "none"  # "none", "brush", "eraser"

        # Cena e visualização
        self.scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.scene)
        self.scene.addPixmap(self.image)

        # Botões de edição
        self.brush_button = QPushButton("Pincel")
        self.brush_button.setCheckable(True)
        self.brush_button.clicked.connect(self.activate_brush)

        self.eraser_button = QPushButton("Borracha")
        self.eraser_button.setCheckable(True)
        self.eraser_button.clicked.connect(self.activate_eraser)

        self.none_button = QPushButton("Desativar Ferramentas")
        self.none_button.clicked.connect(self.deactivate_tools)

        # Layout de botões
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.brush_button)
        button_layout.addWidget(self.eraser_button)
        button_layout.addWidget(self.none_button)

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.graphics_view)
        self.setLayout(layout)

        # Eventos e foco
        self.setFocusPolicy(Qt.StrongFocus)
        self.graphics_view.setFocusPolicy(Qt.StrongFocus)
        self.graphics_view.viewport().installEventFilter(self)

    # === Ferramentas ===
    def activate_brush(self):
        self.edit_mode = "brush"
        self.brush_button.setChecked(True)
        self.eraser_button.setChecked(False)

    def activate_eraser(self):
        self.edit_mode = "eraser"
        self.eraser_button.setChecked(True)
        self.brush_button.setChecked(False)

    def deactivate_tools(self):
        self.edit_mode = "none"
        self.brush_button.setChecked(False)
        self.eraser_button.setChecked(False)

    # === Eventos ===
    def eventFilter(self, obj, event):
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove):
            if event.buttons() & Qt.LeftButton:
                pos = self.graphics_view.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())

                if self.edit_mode in ["brush", "eraser"]:
                    self.modify_mask_at(x, y)
                    return True  # evita propagação

                # modo normal: adiciona ponto e prediz
                self.point_manager.add_point(x, y, 1)
                self.draw_point(x, y, QColor("green"))
                self.predict()
                return True

            elif event.buttons() & Qt.RightButton and self.edit_mode == "none":
                pos = self.graphics_view.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())
                self.point_manager.add_point(x, y, 0)
                self.draw_point(x, y, QColor("red"))
                self.predict()
                return True

        return super().eventFilter(obj, event)

    # === Modificação de Máscara ===
    def modify_mask_at(self, x, y):
        if self.binary_mask is None:
            print("[modify_mask_at] Máscara binária não definida ainda.")
            return

        print(f"[modify_mask_at] Modificando em ({x}, {y}) com modo {self.edit_mode}")

        h, w = self.binary_mask.shape
        radius = 5
        yy, xx = np.ogrid[:h, :w]
        dist = (xx - x) ** 2 + (yy - y) ** 2
        mask_area = dist <= radius**2

        if self.edit_mode == "brush":
            self.binary_mask[mask_area] = 1
        elif self.edit_mode == "eraser":
            self.binary_mask[mask_area] = 0

        self.update_mask_overlay()

    def update_mask_overlay(self):
        if self.binary_mask is None:
            return

        base_image = self.image_loader.last_image_array.copy()
        base_image = self.image_loader.process_image_to_scene(image_np=base_image)

        overlay = base_image.copy()
        overlay[self.binary_mask == 1] = [255, 0, 0]

        alpha = 0.5
        combined = base_image.copy()
        mask_indices = self.binary_mask == 1
        combined[mask_indices] = (
            alpha * overlay[mask_indices] + (1 - alpha) * base_image[mask_indices]
        ).astype(np.uint8)

        h, w, c = combined.shape
        qimage = QImage(combined.data, w, h, combined.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.scene.clear()
        self.scene.addPixmap(pixmap)

        for x, y, label in self.point_manager.points:
            color = QColor("green") if label == 1 else QColor("red")
            self.draw_point(x, y, color)

    # === Pontos ===
    def draw_point(self, x, y, color):
        radius = 5
        ellipse = self.scene.addEllipse(
            x - radius, y - radius, radius * 2, radius * 2, QPen(color), color
        )
        ellipse.setZValue(1)
        self.point_items.append(ellipse)

    def undo_last_point(self):
        removed = self.point_manager.undo_last_point()
        if removed is None:
            return
        if self.point_items:
            item = self.point_items.pop()
            self.scene.removeItem(item)
        self.predict()

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo_last_point()

    # === Predição ===
    def predict(self):
        if (
            self.image is None
            or len(self.point_manager.points) == 0
            or self.image_loader.last_image_array is None
        ):
            self.show_error(
                "Não foi possível carregar a imagem ou os pontos. Analise o terminal para verificar o log corretamente."
            )
            print(20 * "-")
            print("Imagem na cena foi carregado? ", self.image is not None)
            print(
                "Pontos adicionados foram carregados? ",
                len(self.point_manager.points) > 0,
            )
            print(
                "Imagem em np.ndarray foi carregado? ",
                self.image_loader.last_image_array is not None,
            )
            print(20 * "-")

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
            self.scene.clear()  # clean scene
            self.scene.addPixmap(pixmap)
            return

        low_res_logits, masks_logits = self.segmenter.segment(
            image=self.image_loader.last_image_array, points=self.point_manager.points
        )
        masks_logits = masks_logits.detach().cpu()

        base_image = self.image_loader.last_image_array.copy()
        self.best_idx = None

        def on_select(idx, threshold):
            self.best_idx = idx
            self.segmenter.update_prev_low_res_logits(
                low_res_logits[0, idx].unsqueeze(0).unsqueeze(0).to(self.device)
            )
            for w in self.mask_windows:
                w.close()
            self.mask_windows.clear()

            selected_mask = torch.sigmoid(masks_logits[0, idx]).numpy()
            binary_mask = (selected_mask > threshold).astype(np.uint8)
            binary_mask = cv2.resize(
                binary_mask,
                (self.image.width(), self.image.height()),
                interpolation=cv2.INTER_NEAREST,
            )
            self.binary_mask = binary_mask
            self.update_mask_overlay()

        # Fecha janelas anteriores e abre nova
        for w in self.mask_windows:
            w.close()
        self.mask_windows.clear()

        window = HypothesesWindow(
            base_image=base_image,
            masks_logits=masks_logits[0],
            callback_on_select=on_select,
        )
        window.show()
        self.mask_windows.append(window)

    def reset(self):
        self.scene.clear()
        self.scene.addPixmap(self.image)
        self.point_manager.clear()
        self.point_items.clear()
        self.segmenter.clear_prev_low_res_logits()
        self.binary_mask = None
        self.deactivate_tools()
        for w in self.mask_windows:
            w.close()
        self.mask_windows.clear()

    def get_segmentation_data(self):
        return {"points": self.point_manager.points, "mask": self.binary_mask}

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

        self.predict()

    def show_error(self, message):
        print("Erro:", message)
        QMessageBox.warning(None, "Erro", message)
