import sys
import json
import torch
import argparse
import traceback
from pathlib import Path
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from src.ui.main_interface import MainInterface
from src.models.segment_anything.model_instantiator import SAM_Instantiator
from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox


class LoadModelThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)  # novo sinal para capturar erros

    def __init__(
        self,
        conf_path: str = "src/settings/conf.json",
        use_finetune: bool = True,
    ):
        super().__init__()
        self.use_finetune = use_finetune
        try:
            with open(Path(conf_path), "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Erro ao carregar configuração: {str(e)}")

    def run(self):
        try:
            # Simulando o carregamento do modelo
            self.load_model()
            self.finished.emit()  # Emite o sinal quando o carregamento estiver completo
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Erro ao carregar o modelo: {str(e)}")

    def load_model(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print("Carregando modelo no device: ", device)

        model_instantiator = SAM_Instantiator(
            num_classes=self.config["num_classes"],
            vit_type=self.config["vit_type"],
            multimask_output=False,
            apply_adapter=None,
            apply_freeze=None,
            pixel_mean=self.config["pixel_mean"],
            pixel_std=self.config["pixel_std"],
        )
        if not self.use_finetune:
            print("Usando pesos *originais SAM*")
            """ use this for load original SAM weights"""
            self.model = model_instantiator.create_model_and_load_backbone(
                backbone_checkpoint_path=self.config["backbone_checkpoint_path"]
            ).to(device)
        else:
            print("Usando pesos *finetune*")
            """ use this for load finetuned weights """
            self.model = model_instantiator.load_model_from_checkpoint(
                checkpoint_path=self.config["load_model_from_checkpoint"],
                return_prediction_only=False
            ).to(device)


class MainApp(QApplication):
    def __init__(self, sys_argv, use_finetune=True):
        super().__init__(sys_argv)
        self.use_finetune = use_finetune

        try:
            # Criar o splash com o título
            splash_pix = self.create_splash_with_title("src/assets/logo.png", "Minerva Segmenter")
            self.splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
            self.splash.show()

            # Inicializa o carregamento do modelo em um thread separado
            self.model_loader_thread = LoadModelThread(
                conf_path="src/settings/conf.json",
                use_finetune=self.use_finetune
            )
            self.model_loader_thread.progress.connect(self.update_splash_progress)
            self.model_loader_thread.finished.connect(self.on_model_loaded)

            self.model_loader_thread.start()
        except Exception as e:
            self.show_error(f"Erro ao iniciar aplicação: {str(e)}")
    
    def create_splash_with_title(self, image_path, title):
        # Carregar a imagem original
        original_pixmap = QPixmap(image_path)
        if original_pixmap.isNull():
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

        # Redimensionar a imagem original conforme necessário
        scaled_pixmap = original_pixmap.scaled(
            400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Definir a altura adicional para o texto (por exemplo, 50 pixels)
        text_height = 50
        # Criar um novo QPixmap com altura adicional para o texto
        new_width = scaled_pixmap.width()
        new_height = scaled_pixmap.height() + text_height
        new_pixmap = QPixmap(new_width, new_height)
        new_pixmap.fill(Qt.transparent)  # Preencher com transparente (ou outra cor de fundo)

        # Criar um QPainter para desenhar no novo QPixmap
        painter = QPainter(new_pixmap)
        # Desenhar a imagem redimensionada no topo
        painter.drawPixmap(0, 0, scaled_pixmap)

        # Configurar a fonte e a cor do texto
        font = QFont()
        font.setPointSize(14)  # Tamanho da fonte
        font.setBold(True)     # Negrito
        painter.setFont(font)
        painter.setPen(QColor(Qt.white))  # Cor do texto

        # Desenhar o texto abaixo da imagem
        # Centralizar o texto horizontalmente
        text_rect = QRect(0, scaled_pixmap.height(), new_width, text_height)
        painter.drawText(text_rect, Qt.AlignCenter, title)

        # Finalizar o desenho
        painter.end()

        return new_pixmap

    def update_splash_progress(self, progress):
        # Atualiza a barra de progresso na tela de splash
        self.splash.showMessage(
            f"Carregando modelo... {progress}%",
            Qt.AlignBottom | Qt.AlignCenter,
            Qt.white,
        )

    def on_model_loaded(self):
        try:
            # Fecha a tela de splash e mostra a janela principal
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print("Carregando device para interface: ", device)

            self.splash.finish(self.splash)
            self.main_window = MainInterface(
                config=self.model_loader_thread.config, model=self.model_loader_thread.model, device=device
            )
            self.main_window.show()
        except Exception as e:
            traceback.print_exc()
            self.show_error(f"Erro ao carregar interface principal: {str(e)}")

    def show_error(self, message):
        print("Erro:", message)
        QMessageBox.critical(None, "Erro", message)
        self.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_finetune", action="store_true", help="Usar pesos finetunados")
    args = parser.parse_args()

    app = MainApp(sys.argv, use_finetune=args.use_finetune)
    sys.exit(app.exec_())
