import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.colors import ListedColormap
from src.controllers.image_loader import ImageLoader
from src.ui.facies_segmentation_tab import FaciesSegmentationTab
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget,
    QLabel,
)


class MainInterface(QMainWindow):
    def __init__(self, config, model, device):
        super().__init__()
        self.setWindowTitle("Minerva Segmenter - Seismic Facies Segmentation Tool")
        self.setGeometry(100, 100, 2000, 1400)

        self.device = device
        self.config = config
        self.model = model
        self.num_facies = config["num_facies"]

        self.image_loader = ImageLoader()
        self.facies_tabs = []

        # Botões principais
        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.clicked.connect(self.load_image)

        self.save_button = QPushButton("Salvar Segmentação")
        self.save_button.clicked.connect(self.save_segmentation)

        self.reset_button = QPushButton("Reiniciar Segmentação")
        self.reset_button.clicked.connect(self.reset_segmentation)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()

        self.tab_widget = QTabWidget()

        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.tab_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Após criar todas as abas normais
        self.aggregate_button = QPushButton("Atualizar Resultado")
        self.aggregate_button.clicked.connect(self.show_aggregated_result)
        button_layout.addWidget(self.aggregate_button)
        self.result_tab = QLabel()
        self.result_tab.setAlignment(Qt.AlignCenter)

    def load_image(self):
        pixmap = self.image_loader.load_image()
        if not pixmap:
            return

        # Limpa abas anteriores
        self.tab_widget.clear()
        self.facies_tabs.clear()

        # Cria `num_facies` abas com cópias da imagem
        for i in range(self.num_facies):
            tab = FaciesSegmentationTab(
                name=f"Facie {i+1}",
                image=pixmap,
                image_loader=self.image_loader,
                config=self.config,
                model=self.model,
                device=self.device,
            )
            self.tab_widget.addTab(tab, f"Facie {i+1}")
            self.facies_tabs.append(tab)
        # Recria a aba de resultado agregado após limpar o tab_widget
        self.tab_widget.addTab(self.result_tab, "Resultado Agregado")

    def save_segmentation(self):
        # all_data = []
        # for tab in self.facies_tabs:
        #     data = tab.get_segmentation_data()
        #     all_data.append(data)
        # aqui você pode combinar máscaras, salvar etc.
        print("TODO - falta implementar esse método!")

    def aggregate_masks(self):
        if not self.facies_tabs:
            return

        # Assume que todas as máscaras têm o mesmo shape
        mask_shape = None
        aggregated_mask = None

        for idx, tab in enumerate(self.facies_tabs):
            data = tab.get_segmentation_data()
            binary_mask = data.get("mask")  # np.ndarray binária

            if binary_mask is None:
                continue

            if mask_shape is None:
                mask_shape = binary_mask.shape
                aggregated_mask = np.full(
                    mask_shape, fill_value=7, dtype=np.uint8
                )  # 7 = buraco

            # Aplica valor da fácies nos pixels 1
            mask_idx = binary_mask == 1
            aggregated_mask[mask_idx] = idx  # ou idx - 1 se quiser começar do -1

        return aggregated_mask

    def show_aggregated_result(self):
        agg = self.aggregate_masks()
        if agg is None:
            self.result_tab.setText("Nenhuma máscara disponível.")
            return

        # Converte para RGB colorido (opcional)
        label_cmap = ListedColormap(
            [
                [0.294, 0.439, 0.733],  # classe 0
                [0.588, 0.761, 0.867],  # classe 1
                [0.890, 0.965, 0.976],  # classe 2
                [0.980, 0.875, 0.467],  # classe 3
                [0.961, 0.471, 0.294],  # classe 4
                [0.847, 0.157, 0.141],  # classe 5
                [1.000, 0.753, 0.796],  # classe 6 (fundo/buraco)
            ]
        )
        h, w = agg.shape

        # Aplica colormap e converte para RGB uint8
        colored = label_cmap(agg)[:, :, :3]  # valores float [0, 1]
        rgb_image = (colored * 255).astype(np.uint8)

        # Cria QPixmap
        img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        self.result_tab.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def reset_segmentation(self):
        for tab in self.facies_tabs:
            tab.reset()

        # Limpa resultado agregado
        if isinstance(self.result_tab, QLabel):
            self.result_tab.clear()
            self.result_tab.setText("Resultado ainda não gerado.")
            self.result_tab.setAlignment(Qt.AlignCenter)
