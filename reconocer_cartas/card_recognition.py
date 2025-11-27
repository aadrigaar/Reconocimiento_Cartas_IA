import cv2
import numpy as np
import os
import glob

TEMPLATE_WIDTH = 226
TEMPLATE_HEIGHT = 314

class PixelCardRecognizer:

    def __init__(self, template_folder="templates"):
        self.template_folder = template_folder
        self.templates = self.load_templates()
        
        # Diccionario de problemas específicos y sus soluciones
        self.problem_cards = {
            '8_DIAMANTES': self.detect_8_diamonds,
            '2_TREBOLES': self.detect_2_clubs, 
            '3_TREBOLES': self.detect_3_clubs,
            '8_TREBOLES': self.detect_8_clubs
        }

    def load_templates(self):
        """Carga todas las plantillas y las redimensiona exactamente al mismo tamaño"""
        templates = {}
        for suit_folder in glob.glob(os.path.join(self.template_folder, "*")):
            if not os.path.isdir(suit_folder):
                continue
            for img_path in glob.glob(os.path.join(suit_folder, "*.png")):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARNING] No se pudo cargar: {img_path}")
                    continue
                    
                # Convertir RGBA a BGR si es necesario
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # REDIMENSIONAR EXACTAMENTE al tamaño objetivo
                img = cv2.resize(img, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))
                
                # Convertir a grises y normalizar
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                
                key = os.path.splitext(os.path.basename(img_path))[0]
                templates[key] = gray
                print(f"[LOADED] {key} - Shape: {gray.shape}")
        
        print(f"[INFO] {len(templates)} plantillas cargadas correctamente.")
        return templates

    def get_card_contour(self, frame):
        """Detecta contornos de cartas"""
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

        possible_cards.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        return possible_cards

    def warp_card(self, frame, contour):
        """Aplica transformación de perspectiva y redimensiona EXACTAMENTE"""
        pts = contour.reshape(4,2).astype("float32")
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
        
        # Asegurar el tamaño exacto
        warp = cv2.resize(warp, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))
        return warp

    def preprocess_card(self, card_img):
        """Preprocesa la carta para comparación - asegura tamaño y formato"""
        # Asegurar que tiene 3 canales (BGR)
        if len(card_img.shape) == 3 and card_img.shape[2] == 4:
            card_img = cv2.cvtColor(card_img, cv2.COLOR_BGRA2BGR)
        elif len(card_img.shape) == 2:
            card_img = cv2.cvtColor(card_img, cv2.COLOR_GRAY2BGR)
        
        # REDIMENSIONAR EXACTAMENTE al mismo tamaño que las plantillas
        card_img = cv2.resize(card_img, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))
        
        # Convertir a grises y normalizar
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Aplicar suavizado para reducir ruido
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray

    def safe_compare(self, img1, img2):
        """Comparación segura que verifica tamaños"""
        # Verificar dimensiones
        if img1.shape != img2.shape:
            print(f"[ERROR] Shape mismatch: {img1.shape} vs {img2.shape}")
            # Redimensionar ambas al tamaño común
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        return cv2.absdiff(img1, img2)

    def compare_by_pixels(self, card_img):
        """
        Comparación directa por píxeles con todas las rotaciones
        """
        processed_card = self.preprocess_card(card_img)
        
        print(f"[DEBUG] Card shape: {processed_card.shape}")
        
        best_match = "UNKNOWN"
        best_score = float('inf')

        # Generar las 4 rotaciones
        rotations = [
            processed_card,  # 0°
            cv2.rotate(processed_card, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(processed_card, cv2.ROTATE_180),
            cv2.rotate(processed_card, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        # Asegurar que las rotaciones tengan el tamaño correcto
        for i in range(len(rotations)):
            rotations[i] = cv2.resize(rotations[i], (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))

        for rotated in rotations:
            for name, template in self.templates.items():
                # Comparación segura con verificación de tamaño
                diff = self.safe_compare(rotated, template)
                score = np.sum(diff)
                
                if score < best_score:
                    best_score = score
                    best_match = name

        # VERIFICACIÓN ESPECÍFICA PARA CARTAS PROBLEMÁTICAS
        final_match = self.verify_problem_cards(processed_card, best_match, best_score)
        
        return final_match, best_score

    def verify_problem_cards(self, card_img, current_match, current_score):
        """
        Verificación específica para las cartas que fallan
        """
        # Si no es una de las problemáticas, devolver la actual
        if current_match not in self.problem_cards:
            return current_match
        
        # Llamar a la función específica de verificación
        verified_match = self.problem_cards[current_match](card_img)
        
        return verified_match if verified_match else current_match

    # DETECTORES ESPECÍFICOS PARA CADA CARTA PROBLEMÁTICA

    def detect_8_diamonds(self, card_img):
        """Detecta específicamente el 8 de diamantes"""
        # Comparación directa entre las opciones probables
        scores = {}
        candidates = ["8_DIAMANTES", "3_DIAMANTES", "8_TREBOLES", "3_TREBOLES"]
        
        for name in candidates:
            if name in self.templates:
                diff = self.safe_compare(card_img, self.templates[name])
                scores[name] = np.sum(diff)
                print(f"[8_DIAMONDS_CHECK] {name}: {scores[name]}")
        
        return min(scores, key=scores.get) if scores else "8_DIAMANTES"

    def detect_2_clubs(self, card_img):
        """Detecta específicamente el 2 de tréboles"""
        scores = {}
        candidates = ["2_TREBOLES", "3_TREBOLES", "8_TREBOLES"]
        
        for name in candidates:
            if name in self.templates:
                diff = self.safe_compare(card_img, self.templates[name])
                scores[name] = np.sum(diff)
                print(f"[2_CLUBS_CHECK] {name}: {scores[name]}")
        
        return min(scores, key=scores.get) if scores else "2_TREBOLES"

    def detect_3_clubs(self, card_img):
        """Detecta específicamente el 3 de tréboles"""
        return self.detect_2_clubs(card_img)  # Usa la misma lógica

    def detect_8_clubs(self, card_img):
        """Detecta específicamente el 8 de tréboles"""
        scores = {}
        candidates = ["8_TREBOLES", "3_TREBOLES", "8_DIAMANTES"]
        
        for name in candidates:
            if name in self.templates:
                diff = self.safe_compare(card_img, self.templates[name])
                scores[name] = np.sum(diff)
                print(f"[8_CLUBS_CHECK] {name}: {scores[name]}")
        
        return min(scores, key=scores.get) if scores else "8_TREBOLES"

    def debug_comparison(self, card_img):
        """
        Función de debug para ver todas las comparaciones
        """
        processed_card = self.preprocess_card(card_img)
        
        print(f"\n=== COMPARACIÓN DETALLADA ===")
        print(f"Card shape: {processed_card.shape}")
        
        scores = {}
        
        for name, template in self.templates.items():
            diff = self.safe_compare(processed_card, template)
            score = np.sum(diff)
            scores[name] = score

        # Mostrar top 10
        print("\n--- TOP 10 COINCIDENCIAS ---")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        for i, (name, score) in enumerate(sorted_scores[:10]):
            print(f"{i+1}. {name}: {score}")
        
        # Mostrar específicamente las problemáticas
        print("\n--- CARTAS PROBLEMÁTICAS ---")
        problem_cards = ["8_DIAMANTES", "2_TREBOLES", "3_TREBOLES", "8_TREBOLES", 
                        "3_DIAMANTES", "2_DIAMANTES"]
        for card in problem_cards:
            if card in scores:
                print(f"{card}: {scores[card]}")
        
        return sorted_scores[0][0]  # Devolver la mejor coincidencia

    def run(self, debug_mode=False):
        """Ejecuta el reconocimiento"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara")
            return

        print("[INFO] Cámara iniciada. Presiona 'q' para salir, 'd' para debug")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo leer el frame")
                continue

            card_contours = self.get_card_contour(frame)
            detected_cards = []

            for idx, card_contour in enumerate(card_contours):
                try:
                    warp = self.warp_card(frame, card_contour)
                    
                    if debug_mode and idx == 0:
                        match_name = self.debug_comparison(warp)
                        score = 0
                    else:
                        match_name, score = self.compare_by_pixels(warp)
                    
                    detected_cards.append(match_name)

                    cv2.drawContours(frame, [card_contour], -1, (0,255,0), 3)

                    M = cv2.moments(card_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, f"{match_name} ({score:.0f})", (cX - 70, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    cv2.imshow(f"Carta {idx+1}", warp)
                    
                except Exception as e:
                    print(f"[ERROR] Procesando carta {idx}: {e}")
                    continue

            cv2.putText(frame, f"Cartas: {len(detected_cards)}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            cv2.imshow("Reconocimiento por Píxeles", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[DEBUG] Modo debug: {debug_mode}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = PixelCardRecognizer()
    print("[INFO] Presiona 'd' para activar modo debug")
    recognizer.run(debug_mode=False)