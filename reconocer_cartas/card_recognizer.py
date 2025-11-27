import cv2
import numpy as np
import os
import glob

TEMPLATE_WIDTH = 226
TEMPLATE_HEIGHT = 314

class CardRecognizer:

    def __init__(self, template_folder="templates"):
        self.template_folder = template_folder
        self.templates = self.load_templates()

    def load_templates(self):
        """
        Carga TODAS las plantillas desde templates/*/*.png
        Las redimensiona a 226x314
        """
        templates = {}

        for suit_folder in glob.glob(os.path.join(self.template_folder, "*")):
            if not os.path.isdir(suit_folder):
                continue

            for img_path in glob.glob(os.path.join(suit_folder, "*.png")):
                img = cv2.imread(img_path)

                if img is None:
                    continue

                # IMPORTANTE: Convierte RGBA a BGR si tiene 4 canales
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                img = cv2.resize(img, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))

                key = os.path.splitext(os.path.basename(img_path))[0]  # ej: "6_PICAS"
                templates[key] = img

        print(f"[INFO] {len(templates)} plantillas cargadas correctamente.")
        return templates

    def get_card_contour(self, frame):
        """
        Busca TODOS los contornos con forma de carta (cuadriláteros)
        Devuelve una lista de contornos ordenados por área
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        possible_cards = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.contourArea(approx) > 5000:
                possible_cards.append(approx)

        if len(possible_cards) == 0:
            return []

        # Ordena por área (mayor a menor)
        possible_cards.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        return possible_cards

    def warp_card(self, frame, contour):
        """
        Aplica transformación de perspectiva a la carta
        y la deja EXACTAMENTE en 226x314 píxeles
        """
        pts = contour.reshape(4,2).astype("float32")

        # Ordena esquinas (top-left, top-right, bottom-right, bottom-left)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect = np.zeros((4,2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        dst = np.array([
            [0, 0],
            [TEMPLATE_WIDTH-1, 0],
            [TEMPLATE_WIDTH-1, TEMPLATE_HEIGHT-1],
            [0, TEMPLATE_HEIGHT-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(frame, M, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))

        return warp

    def compare_cards(self, card_img):
        """
        Compara la carta detectada contra todas las plantillas cargadas.
        Prueba 4 rotaciones (0°, 90°, 180°, 270°) para reconocer cartas giradas.
        Compara en COLOR (BGR) para diferenciar palos correctamente.
        Devuelve el nombre con menor error.
        """
        best_match = "UNKNOWN"
        best_score = float('inf')

        # Asegura que la carta tenga 3 canales (BGR)
        if len(card_img.shape) == 3 and card_img.shape[2] == 4:
            card_img = cv2.cvtColor(card_img, cv2.COLOR_BGRA2BGR)

        # Genera las 4 rotaciones de la carta
        rot_90 = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)
        rot_90 = cv2.resize(rot_90, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))
        
        rot_180 = cv2.rotate(card_img, cv2.ROTATE_180)
        
        rot_270 = cv2.rotate(card_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_270 = cv2.resize(rot_270, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))
        
        rotations = [
            card_img,   # 0°
            rot_90,     # 90°
            rot_180,    # 180°
            rot_270     # 270°
        ]

        # Compara cada rotación con todas las plantillas
        for rotated in rotations:
            for name, tmpl in self.templates.items():
                # Comparación directa en color
                diff = cv2.absdiff(rotated, tmpl)
                score = np.sum(diff)

                if score < best_score:
                    best_score = score
                    best_match = name

        return best_match, best_score

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Obtiene TODAS las cartas detectadas
            card_contours = self.get_card_contour(frame)

            detected_cards = []

            # Procesa cada carta encontrada
            for idx, card_contour in enumerate(card_contours):
                warp = self.warp_card(frame, card_contour)

                match_name, score = self.compare_cards(warp)

                detected_cards.append(match_name)

                # Dibuja el contorno
                cv2.drawContours(frame, [card_contour], -1, (0,255,0), 3)

                # Calcula el centro del contorno para poner el texto
                M = cv2.moments(card_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, match_name, (cX - 70, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # Muestra cada carta detectada en ventana separada
                cv2.imshow(f"Carta detectada {idx+1} ({match_name})", warp)

            # Muestra el número de cartas detectadas
            cv2.putText(frame, f"Cartas: {len(detected_cards)}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            cv2.imshow("Reconocimiento de cartas", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = CardRecognizer()
    recognizer.run()