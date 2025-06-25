class PointManager:
    def __init__(self):
        self.points = []

    def add_point(self, x: int, y: int, label: int):
        self.points.append((x, y, label))
        print(f"[ADD] Ponto adicionado: ({x}, {y}, {label})")

    def count_points(self):
        pos = sum(1 for _, _, label in self.points if label == 1)
        neg = sum(1 for _, _, label in self.points if label == 0)
        return pos, neg

    def clear(self):
        self.points.clear()
    
    def undo_last_point(self):
        if self.points:
            p = self.points.pop()
            print(f"[UNDO] Ponto removido: {p}")
            return p
        print("[UNDO] Nenhum ponto para remover.")
        return None