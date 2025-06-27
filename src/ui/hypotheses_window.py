from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QScrollArea,
    QApplication,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QEvent
import numpy as np
import torch
import cv2


class HypothesesWindow(QWidget):
    def __init__(self, base_image, masks_logits, callback_on_select):
        super().__init__()
        self.setWindowTitle("Escolha uma hipótese")
        self.resize(1500, 900)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.frames = []
        for idx in range(masks_logits.shape[0]):
            frame = SingleHypothesisFrame(
                index=idx,
                base_image_np=base_image,
                mask_logit_tensor=masks_logits[idx],
                callback_on_select=callback_on_select,
            )
            self.layout.addWidget(frame)
            self.frames.append(frame)


class SingleHypothesisFrame(QWidget):
    def __init__(self, index, base_image_np, mask_logit_tensor, callback_on_select):
        super().__init__()
        self.index = index
        self.original_image = base_image_np
        self.mask_logit = mask_logit_tensor.squeeze().cpu()
        self.threshold = 0.5
        self.callback_on_select = callback_on_select
        self.zoom = 1.0

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.viewport().installEventFilter(self)

        self.title_label = QLabel()
        self.update_title()  # Inicializa o texto

        self.select_button = QPushButton("Selecionar")
        self.select_button.clicked.connect(self.select_mask)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.select_button)
        self.setLayout(layout)

        self.update_overlay()

    def update_title(self):
        self.title_label.setText(
            f"Hipótese {self.index} - Threshold: {self.threshold:.1f}"
        )

    def update_overlay(self):
        # Atualiza título
        self.update_title()

        # Gera máscara
        mask_prob = torch.sigmoid(self.mask_logit)
        binary_mask = (mask_prob > self.threshold).numpy().astype(np.uint8)

        resized_mask = cv2.resize(
            binary_mask,
            (self.original_image.shape[1], self.original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Normaliza imagem se necessário
        base_image = self.original_image.copy()
        if base_image.dtype != np.uint8:
            base_image = base_image.astype(np.float32)
            base_image -= base_image.min()
            base_image /= max(base_image.max(), 1e-5)
            base_image *= 255
            base_image = base_image.astype(np.uint8)

        if base_image.ndim == 2:
            base_image = np.stack([base_image] * 3, axis=-1)

        # Cria overlay transparente somente nas regiões da máscara
        overlay = base_image.copy()
        overlay[resized_mask == 1] = [255, 0, 0]

        alpha = 0.5
        combined = base_image.copy()
        mask_indices = resized_mask == 1
        combined[mask_indices] = (
            alpha * overlay[mask_indices] + (1 - alpha) * base_image[mask_indices]
        ).astype(np.uint8)

        # Aplica zoom
        if self.zoom != 1.0:
            combined = cv2.resize(
                combined,
                (0, 0),
                fx=self.zoom,
                fy=self.zoom,
                interpolation=cv2.INTER_LINEAR,
            )

        h, w, c = combined.shape
        qimage = QImage(combined.data, w, h, combined.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

    def select_mask(self):
        self.callback_on_select(self.index, self.threshold)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            modifiers = QApplication.keyboardModifiers()

            delta = event.angleDelta().y() // 120

            if modifiers & Qt.AltModifier:
                # Ajusta threshold com passos variáveis
                new_threshold = self.threshold

                for _ in range(abs(delta)):
                    step = 0.01 if 0 < new_threshold < 0.1 else 0.05
                    if delta > 0:
                        new_threshold += step
                    else:
                        new_threshold -= step

                # Garante que o threshold não vá abaixo de zero
                new_threshold = np.clip(new_threshold, 0.001, 1.0)  # evita exatamente 0
                self.threshold = round(new_threshold, 3)
                self.update_overlay()
                return True

            elif modifiers & (
                Qt.ControlModifier | Qt.MetaModifier
            ):  # Ctrl no Windows/Linux, ⌘ no Mac
                # Ajusta zoom
                self.zoom = np.clip(self.zoom + delta * 0.1, 0.2, 3.0)
                self.update_overlay()
                return True

        return super().eventFilter(obj, event)
