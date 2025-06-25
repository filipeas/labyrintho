import numpy as np
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from minerva.data.readers import TiffReader, PNGReader


class ImageLoader:
    def __init__(self):
        self.last_image_array: Optional[np.ndarray] = None

    def load_image(self) -> QPixmap:
        path_str, _ = QFileDialog.getOpenFileName(
            None, "Abrir Imagem", "", "Imagens (*.png *.tif *.tiff)"
        )
        if not path_str:
            return None

        path = Path(path_str)
        ext = path.suffix.lower()

        try:
            image_np = None

            if ext in [".tif", ".tiff"]:
                reader = TiffReader(path.parent)
                idx = reader.files.index(path)
                image_np = reader[idx]

            elif ext == ".png":
                reader = PNGReader(path.parent)
                idx = reader.files.index(path)
                image_np = reader[idx]

            elif ext in [".jpg", ".jpeg"]:
                raise NotImplemented("Imagem JPG não está implementado ainda.")
                # pixmap = QPixmap(str(path))
                # if pixmap.isNull():
                #     raise ValueError("Erro ao carregar imagem JPG.")
                # return pixmap
            else:
                raise ValueError("Formato não suportado.")

            self.last_image_array = image_np.copy()

            image_vis = self.process_image_to_scene(image_np)

            h, w, ch = image_vis.shape
            bytes_per_line = ch * w
            qimage = QImage(image_vis.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qimage)

        except Exception as e:
            print(f"[Erro ao carregar imagem com Minerva]: {e}")
            return None

    def process_image_to_scene(self, image_np):
        # only for presentation: convert to uint8 RGB
        image_vis = image_np
        if image_vis.dtype != np.uint8:
            image_vis = image_vis.astype(np.float32)
            image_vis -= image_vis.min()
            image_vis /= image_vis.max()
            image_vis *= 255
            image_vis = image_vis.astype(np.uint8)

        if image_vis.ndim == 2:
            image_vis = np.stack([image_vis] * 3, axis=-1)
        return image_vis
