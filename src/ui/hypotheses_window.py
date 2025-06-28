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
    def __init__(
        self,
        base_image,
        masks_logits,
        callback_on_select,
        on_cancel,
        points=None,
        prev_mask=None,
    ):
        super().__init__()
        self.setWindowTitle("Escolha uma hipótese")
        self.resize(2200, 1400)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.points = points or []
        self.prev_mask = prev_mask

        self.frame_layout = QHBoxLayout()
        self.frames = []
        for idx in range(masks_logits.shape[0]):
            frame = SingleHypothesisFrame(
                index=idx,
                base_image_np=base_image,
                mask_logit_tensor=masks_logits[idx],
                callback_on_select=callback_on_select,
                points=self.points,
                prev_mask=self.prev_mask,
            )
            self.layout.addWidget(frame)
            self.frames.append(frame)

        # Botão de cancelamento
        self.cancel_button = QPushButton("Cancelar seleção de hipótese")
        self.cancel_button.clicked.connect(on_cancel)

        self.layout.addLayout(self.frame_layout)
        self.layout.addWidget(self.cancel_button)


class SingleHypothesisFrame(QWidget):
    def __init__(
        self,
        index,
        base_image_np,
        mask_logit_tensor,
        callback_on_select,
        points=None,
        prev_mask=None,
    ):
        super().__init__()
        self.index = index
        self.original_image = base_image_np
        self.mask_logit = mask_logit_tensor.squeeze().cpu()
        self.threshold = 0.5
        self.callback_on_select = callback_on_select
        self.zoom = 1.0

        self.points = points or []
        self.prev_mask = prev_mask

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
        self.update_title()

        # Gera máscara atual
        mask_prob = torch.sigmoid(self.mask_logit)
        binary_mask = (mask_prob > self.threshold).numpy().astype(np.uint8)

        resized_mask = cv2.resize(
            binary_mask,
            (self.original_image.shape[1], self.original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        base_image = self.original_image.copy()
        if base_image.dtype != np.uint8:
            base_image = base_image.astype(np.float32)
            base_image -= base_image.min()
            base_image /= max(base_image.max(), 1e-5)
            base_image *= 255
            base_image = base_image.astype(np.uint8)

        if base_image.ndim == 2:
            base_image = np.stack([base_image] * 3, axis=-1)

        # Começa com cópia da imagem
        combined = base_image.copy()

        # 1. Aplica máscara anterior em branco (transparente)
        if self.prev_mask is not None:
            prev_resized = cv2.resize(
                self.prev_mask,
                (base_image.shape[1], base_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            white_overlay = np.full_like(base_image, 255)
            alpha_white = 0.3
            mask_indices = prev_resized == 1
            combined[mask_indices] = (
                alpha_white * white_overlay[mask_indices]
                + (1 - alpha_white) * combined[mask_indices]
            ).astype(np.uint8)

        # 2. Aplica máscara atual em vermelho (transparente)
        red_overlay = np.zeros_like(base_image)
        red_overlay[:, :, 0] = 255  # R=255
        alpha_red = 0.5
        mask_indices = resized_mask == 1
        combined[mask_indices] = (
            alpha_red * red_overlay[mask_indices]
            + (1 - alpha_red) * combined[mask_indices]
        ).astype(np.uint8)

        # 3. Desenha pontos por cima
        for x, y, label in self.points:
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            radius = 4
            cv2.circle(combined, (int(x), int(y)), radius, color, thickness=-1)

        # 4. Aplica zoom se necessário
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
