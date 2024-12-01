import sys
from PyQt5.QtWidgets import QApplication
from src.interfaces.main_interface import MainInterface

class MainApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.main_window = MainInterface()
        self.main_window.show()

if __name__ == "__main__":
    app = MainApp(sys.argv)
    sys.exit(app.exec_())