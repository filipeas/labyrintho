import sys
import json
import torch
import traceback
from pathlib import Path
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
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
    ):
        super().__init__()
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
        # self.model = model_instantiator.create_model_and_load_backbone(
        #     backbone_checkpoint_path=self.config["backbone_checkpoint_path"]
        # ).to(device)
        self.model = model_instantiator.load_model_from_checkpoint(
            checkpoint_path=self.config["load_model_from_checkpoint"],
            return_prediction_only=False
        ).to(device)


class MainApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)

        try:
            splash_pix = QPixmap("src/assets/logo.png").scaled(
                400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
            self.splash.show()

            # Inicializa o carregamento do modelo em um thread separado
            self.model_loader_thread = LoadModelThread(
                conf_path="src/settings/conf.json"
            )
            self.model_loader_thread.progress.connect(self.update_splash_progress)
            self.model_loader_thread.finished.connect(self.on_model_loaded)

            self.model_loader_thread.start()
        except Exception as e:
            self.show_error(f"Erro ao iniciar aplicação: {str(e)}")

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
    app = MainApp(sys.argv)
    sys.exit(app.exec_())
