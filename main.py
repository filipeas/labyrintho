import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplashScreen, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from src.interfaces.main_interface import MainInterface
from src.models.segment_anything import sam_model_registry, SamPredictor
import torch
import time

class LoadModelThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, checkpoint:str="sam_vit_h_4b8939.pth", model_type:str="vit_h", model:str="sam"):
        super().__init__()
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.model = model

    def run(self):
        # Simulando o carregamento do modelo
        self.load_model()
        self.finished.emit()  # Emite o sinal quando o carregamento estiver completo

    def load_model(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print("device: ", device)

        # simulando tempo de carregamento
        for i in range(100):
            self.progress.emit(i + 1)  # Emite progresso de 0 a 100
            time.sleep(0.05)  # Simula o tempo de carregamento

        if self.model == 'sam':
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
        else:
            print("Informe qual modelo deseja executar")
            return

class MainApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        
        splash_pix = QPixmap('src/assets/logo.jpeg').scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)  
        self.splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        self.splash.show()

        # Inicializa o carregamento do modelo em um thread separado
        self.model_loader_thread = LoadModelThread(
            checkpoint='src/models/segment_anything/checkpoints/sam_vit_b_01ec64.pth',
            model_type='vit_b',
            model='sam'
        )
        self.model_loader_thread.progress.connect(self.update_splash_progress)
        self.model_loader_thread.finished.connect(self.on_model_loaded)

        self.model_loader_thread.start()

    def update_splash_progress(self, progress):
        # Atualiza a barra de progresso na tela de splash
        self.splash.showMessage(f'Carregando modelo... {progress}%', Qt.AlignBottom | Qt.AlignCenter, Qt.white)

    def on_model_loaded(self):
        # Fecha a tela de splash e mostra a janela principal
        self.splash.finish(self.splash)
        self.main_window = MainInterface(model=self.model_loader_thread.predictor)
        self.main_window.show()

if __name__ == "__main__":
    app = MainApp(sys.argv)
    sys.exit(app.exec_())