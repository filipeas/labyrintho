from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QApplication,
)
from PyQt5.QtGui import QImage, QPixmap, QWheelEvent
from PyQt5.QtCore import Qt, QObject, QEvent
import numpy as np
import torch
import cv2


class MaskSelectionWindow(QWidget):
    def __init__(self, index, base_image_np, mask_logit_tensor, callback_on_select):
        super().__init__()
        self.index = index
        self.original_image = base_image_np  # shape (H, W, C)
        self.mask_logit = mask_logit_tensor.squeeze().cpu()
        self.threshold = 0.5
        self.callback_on_select = callback_on_select

        self.setWindowTitle(f"Hipótese {index} - Threshold: {self.threshold:.1f}")
        self.resize(600, 600)  # tamanho inicial

        # QLabel com política de redimensionamento
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)  # <- redimensiona imagem

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)

        # Instala filtro de evento no scroll area
        self.scroll_area.viewport().installEventFilter(self)

        self.select_button = QPushButton("Selecionar esta máscara")
        self.select_button.clicked.connect(self.select_mask)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.select_button)
        self.setLayout(layout)

        self.update_overlay()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # Intercepta o scroll no viewport
        if event.type() == QEvent.Type.Wheel:
            wheel_event = QWheelEvent(event)
            modifiers = QApplication.keyboardModifiers()

            if modifiers & Qt.AltModifier:
                # Scroll com Alt → muda threshold, bloqueia scroll
                steps = wheel_event.angleDelta().y() // 120
                if steps != 0:
                    self.threshold = round(
                        np.clip(self.threshold + 0.1 * steps, 0.0, 1.0), 1
                    )
                    self.update_overlay()
                return True  # ← impede propagação para o scroll

        return super().eventFilter(obj, event)

    def update_overlay(self):
        # Aplica sigmoid + threshold
        mask_prob = torch.sigmoid(self.mask_logit)
        binary_mask = (mask_prob > self.threshold).numpy().astype(np.uint8)

        mask_resized = cv2.resize(
            binary_mask,
            (self.original_image.shape[1], self.original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        overlay = self.original_image.copy()
        overlay[mask_resized == 1] = [255, 0, 0]

        alpha = 0.5
        combined = (alpha * overlay + (1 - alpha) * self.original_image).astype(
            np.uint8
        )

        # Converte imagem para QPixmap
        h, w, c = combined.shape
        qimage = QImage(combined.data, w, h, combined.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.setWindowTitle(f"Hipótese {self.index} - Threshold: {self.threshold:.1f}")

    def select_mask(self):
        self.callback_on_select(self.index, self.threshold)
        self.close()
