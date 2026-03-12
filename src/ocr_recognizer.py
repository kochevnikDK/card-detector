import cv2
import numpy as np
import pytesseract
import re
import os
from src.template_matcher import TemplateMatcher

class OCRCardRecognizer:
    def __init__(self, tesseract_path=None, use_templates=True, templates_path="templates"):
        """
        Инициализация OCR распознавателя карт
        
        Args:
            tesseract_path: путь к tesseract.exe (для Windows)
            use_templates: использовать ли шаблоны для распознавания
            templates_path: путь к папке с шаблонами
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Инициализация матчера шаблонов
        self.use_templates = use_templates
        if use_templates and os.path.exists(templates_path):
            self.template_matcher = TemplateMatcher(templates_path)
        else:
            self.template_matcher = None
        
        # Параметры для настройки распознавания
        self.params = {
            # Параметры выделения угла
            'corner_padding': 10,
            'corner_size_factor': 4,  # 1/4 от размера карты
            'corner_zoom_factor': 2,   # Увеличение угла
            
            # Параметры сегментации
            'rank_vertical_split': 0.5,  # 50% сверху для ранга
            'suit_vertical_split': 0.5,  # 50% снизу для масти
            
            # Параметры предобработки
            'preprocess_threshold_block': 11,
            'preprocess_threshold_c': 2,
            'preprocess_gaussian_blur': 3,
            'preprocess_morphology_kernel': 2,
            'preprocess_use_clahe': True,
            'preprocess_clahe_clip': 2.0,
            'preprocess_clahe_grid': 8,
            
            # Параметры OCR для ранга
            'ocr_rank_psm': 8,
            'ocr_rank_oem': 3,
            'ocr_rank_whitelist': '0123456789AJQK',
            
            # Параметры OCR для масти
            'ocr_suit_psm': 8,
            'ocr_suit_oem': 3,
            
            # Параметры цветового анализа
            'color_red_threshold': 100,
            'color_red_percent': 0.1,
            
            # Параметры формы
            'shape_heart_solidity': 0.85,
            'shape_spade_center': 0.4,
            
            # Параметры визуализации
            'show_corner': True,
            'show_roi': True,
            'roi_color_rank': (255, 0, 0),    # Синий для ранга
            'roi_color_suit': (0, 255, 0),    # Зеленый для масти
            'roi_color_corner': (0, 255, 255), # Желтый для угла
            'roi_line_thickness': 2,
            'show_text': True,
            'text_color': (255, 255, 255),
            'text_bg_color': (0, 0, 0),
            'text_scale': 0.5,
            'text_thickness': 1
        }
        
        # Отображение распознанных символов в стандартные обозначения
        self.rank_mapping = {
            'A': 'A', 'ACE': 'A',
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 
            '10': '10',
            'J': 'J', 'JACK': 'J',
            'Q': 'Q', 'QUEEN': 'Q',
            'K': 'K', 'KING': 'K'
        }
        
        # Отображение символов мастей
        self.suit_unicode = {
            'hearts': '♥',
            'diamonds': '♦',
            'clubs': '♣',
            'spades': '♠'
        }
        
        # Отображение имен мастей в символы
        self.suit_names = {
            'hearts': 'hearts',
            'diamonds': 'diamonds',
            'clubs': 'clubs',
            'spades': 'spades'
        }
        
        # Цветовые диапазоны для мастей в HSV
        self.suit_colors = {
            'hearts': {'lower': [0, 50, 50], 'upper': [10, 255, 255]},  # Красный 1
            'diamonds': {'lower': [160, 50, 50], 'upper': [180, 255, 255]},  # Красный 2
            'spades': {'lower': [0, 0, 0], 'upper': [180, 255, 80]},  # Черный
            'clubs': {'lower': [0, 0, 0], 'upper': [180, 255, 80]}  # Черный
        }
    
    def set_params(self, params):
        """Обновление параметров распознавателя"""
        self.params.update(params)
    
    def extract_card_corner(self, image, contour, padding=None):
        """
        Извлечение угла карты с мастью и рангом с визуализацией
        
        Args:
            image: исходное изображение
            contour: контур карты
            padding: отступы от края
            
        Returns:
            corner: изображение угла карты
            warped: выпрямленная карта
            visualization: изображение с отмеченными областями
        """
        try:
            if padding is None:
                padding = self.params['corner_padding']
            
            # Получаем повернутый прямоугольник
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Создаем визуализацию
            vis_image = image.copy()
            
            # Рисуем оригинальный контур
            cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
            
            # Рисуем повернутый прямоугольник
            cv2.drawContours(vis_image, [box], 0, (255, 0, 0), 2)
            
            # Определяем размеры
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Корректируем угол
            angle = rect[2]
            if width < height:
                width, height = height, width
                angle = rect[2] + 90
            
            # Получаем матрицу перспективного преобразования
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], 
                              dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            # Вырезаем верхний левый угол
            corner_size = min(width, height) // self.params['corner_size_factor']
            
            # Координаты угла на исходном изображении для визуализации
            corner_pts = np.array([
                [padding, padding],
                [corner_size+padding, padding],
                [corner_size+padding, corner_size+padding],
                [padding, corner_size+padding]
            ], dtype=np.float32)
            
            # Трансформируем координаты угла обратно на исходное изображение
            invM = cv2.getPerspectiveTransform(dst_pts, src_pts)
            corner_pts_transformed = cv2.perspectiveTransform(corner_pts.reshape(-1, 1, 2), invM)
            corner_pts_transformed = np.int32(corner_pts_transformed)
            
            # Рисуем область угла на исходном изображении
            cv2.polylines(vis_image, [corner_pts_transformed], True, self.params['roi_color_corner'], 
                         self.params['roi_line_thickness'])
            
            # Добавляем текст
            if self.params['show_text']:
                # Находим центр области угла
                M = cv2.moments(corner_pts_transformed.reshape(-1, 2))
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Рисуем текст с фоном
                    text = "Corner ROI"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                          self.params['text_scale'], 
                                                          self.params['text_thickness'])
                    cv2.rectangle(vis_image, (cx - text_w//2 - 5, cy - text_h - 5),
                                (cx + text_w//2 + 5, cy + 5), self.params['text_bg_color'], -1)
                    cv2.putText(vis_image, text, (cx - text_w//2, cy - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale'], 
                               self.params['text_color'], self.params['text_thickness'])
            
            # Вырезаем угол
            corner = warped[padding:corner_size+padding, padding:corner_size+padding]
            
            # Увеличиваем для лучшего OCR
            corner = cv2.resize(corner, None, fx=self.params['corner_zoom_factor'], 
                               fy=self.params['corner_zoom_factor'], interpolation=cv2.INTER_CUBIC)
            
            return corner, warped, vis_image
            
        except Exception as e:
            print(f"Ошибка извлечения угла: {e}")
            return None, None, image
    
    def preprocess_for_ocr(self, image):
        """
        Многоступенчатая предобработка для OCR с использованием параметров
        """
        if image is None or image.size == 0:
            return []
        
        processed_images = []
        
        # 1. Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. Усиление контраста (опционально)
        if self.params['preprocess_use_clahe']:
            clahe = cv2.createCLAHE(clipLimit=self.params['preprocess_clahe_clip'], 
                                    tileGridSize=(self.params['preprocess_clahe_grid'], 
                                                 self.params['preprocess_clahe_grid']))
            gray = clahe.apply(gray)
        
        # 3. Несколько вариантов пороговой обработки
        
        # Вариант A: Адаптивная пороговая
        thresh1 = cv2.adaptiveThreshold(gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        self.params['preprocess_threshold_block'],
                                        self.params['preprocess_threshold_c'])
        processed_images.append(('adaptive', thresh1))
        
        # Вариант B: Otsu's thresholding
        _, thresh2 = cv2.threshold(gray, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('otsu', thresh2))
        
        # Вариант C: Инвертированный
        thresh3 = cv2.bitwise_not(thresh1)
        processed_images.append(('inverted', thresh3))
        
        # Вариант D: Морфологические операции
        kernel = np.ones((self.params['preprocess_morphology_kernel'], 
                         self.params['preprocess_morphology_kernel']), np.uint8)
        thresh4 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        thresh4 = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)
        processed_images.append(('morphology', thresh4))
        
        return processed_images
    
    def recognize_rank(self, corner, roi_vis_image=None, card_index=0, card_center=None):
        """
        Распознавание ранга карты с визуализацией ROI
        """
        if corner is None:
            return None, roi_vis_image
        
        # Разделяем угол на две части
        h, w = corner.shape[:2]
        rank_roi = corner[0:int(h * self.params['rank_vertical_split']), 0:w]
        
        # Визуализация ROI ранга на изображении
        if roi_vis_image is not None and card_center is not None and self.params['show_roi']:
            # Масштабируем координаты обратно
            scale_factor = 1.0 / self.params['corner_zoom_factor']
            roi_h, roi_w = rank_roi.shape[:2]
            roi_w = int(roi_w * scale_factor)
            roi_h = int(roi_h * scale_factor)
            
            # Координаты ROI относительно угла карты
            x_offset = self.params['corner_padding']
            y_offset = self.params['corner_padding']
            
            # Рисуем прямоугольник для ранга
            pts = np.array([
                [card_center[0] - roi_w//2, card_center[1] - roi_h - 20],
                [card_center[0] + roi_w//2, card_center[1] - roi_h - 20],
                [card_center[0] + roi_w//2, card_center[1] - 20],
                [card_center[0] - roi_w//2, card_center[1] - 20]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            cv2.polylines(roi_vis_image, [pts], True, self.params['roi_color_rank'], 
                         self.params['roi_line_thickness'])
            
            # Добавляем текст
            if self.params['show_text']:
                text = f"Rank ROI (Card {card_index+1})"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      self.params['text_scale']*0.8, 
                                                      self.params['text_thickness'])
                cv2.rectangle(roi_vis_image, 
                            (card_center[0] - text_w//2 - 5, card_center[1] - roi_h - 40 - text_h),
                            (card_center[0] + text_w//2 + 5, card_center[1] - roi_h - 40),
                            self.params['text_bg_color'], -1)
                cv2.putText(roi_vis_image, text, 
                           (card_center[0] - text_w//2, card_center[1] - roi_h - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale']*0.8, 
                           self.params['text_color'], self.params['text_thickness'])
        
        # OCR распознавание
        processed_images = self.preprocess_for_ocr(rank_roi)
        
        config = f'--oem {self.params["ocr_rank_oem"]} --psm {self.params["ocr_rank_psm"]} -c tessedit_char_whitelist={self.params["ocr_rank_whitelist"]}'
        
        best_result = None
        best_confidence = 0
        
        for img_name, img in processed_images:
            try:
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                
                for i, text in enumerate(data['text']):
                    if text.strip():
                        text = text.strip().upper()
                        conf = int(data['conf'][i])
                        
                        for pattern, rank in self.rank_mapping.items():
                            if pattern in text or text in pattern:
                                if conf > best_confidence:
                                    best_confidence = conf
                                    best_result = rank
            except Exception:
                continue
        
        # Шаблоны
        if self.use_templates and self.template_matcher:
            template_rank, template_score = self.template_matcher.match_rank(rank_roi)
            if template_rank and template_score * 100 > best_confidence:
                best_result = template_rank
                best_confidence = template_score * 100
        
        return best_result, roi_vis_image
    
    def recognize_suit(self, corner, roi_vis_image=None, card_index=0, card_center=None):
        """
        Распознавание масти карты с визуализацией ROI
        """
        if corner is None:
            return None, roi_vis_image
        
        # Разделяем угол на две части
        h, w = corner.shape[:2]
        suit_roi = corner[int(h * self.params['rank_vertical_split']):h, 0:w]
        
        # Визуализация ROI масти на изображении
        if roi_vis_image is not None and card_center is not None and self.params['show_roi']:
            # Масштабируем координаты обратно
            scale_factor = 1.0 / self.params['corner_zoom_factor']
            roi_h, roi_w = suit_roi.shape[:2]
            roi_w = int(roi_w * scale_factor)
            roi_h = int(roi_h * scale_factor)
            
            # Рисуем прямоугольник для масти
            pts = np.array([
                [card_center[0] - roi_w//2, card_center[1] - 20],
                [card_center[0] + roi_w//2, card_center[1] - 20],
                [card_center[0] + roi_w//2, card_center[1] + roi_h - 20],
                [card_center[0] - roi_w//2, card_center[1] + roi_h - 20]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            cv2.polylines(roi_vis_image, [pts], True, self.params['roi_color_suit'], 
                         self.params['roi_line_thickness'])
            
            # Добавляем текст
            if self.params['show_text']:
                text = f"Suit ROI (Card {card_index+1})"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      self.params['text_scale']*0.8, 
                                                      self.params['text_thickness'])
                cv2.rectangle(roi_vis_image, 
                            (card_center[0] - text_w//2 - 5, card_center[1] - 40 - text_h),
                            (card_center[0] + text_w//2 + 5, card_center[1] - 40),
                            self.params['text_bg_color'], -1)
                cv2.putText(roi_vis_image, text, 
                           (card_center[0] - text_w//2, card_center[1] - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale']*0.8, 
                           self.params['text_color'], self.params['text_thickness'])
        
        # OCR распознавание
        processed_images = self.preprocess_for_ocr(suit_roi)
        config = f'--oem {self.params["ocr_suit_oem"]} --psm {self.params["ocr_suit_psm"]}'
        
        for img_name, img in processed_images:
            try:
                text = pytesseract.image_to_string(img, config=config)
                
                if '♥' in text or '❤' in text:
                    return 'hearts', roi_vis_image
                elif '♦' in text:
                    return 'diamonds', roi_vis_image
                elif '♣' in text:
                    return 'clubs', roi_vis_image
                elif '♠' in text:
                    return 'spades', roi_vis_image
            except:
                continue
        
        # Шаблоны
        if self.use_templates and self.template_matcher:
            template_suit, template_score = self.template_matcher.match_suit(suit_roi)
            if template_suit:
                return template_suit, roi_vis_image
        
        # Цветовой анализ
        suit_by_color = self.recognize_suit_by_color(suit_roi)
        if suit_by_color:
            return suit_by_color, roi_vis_image
        
        # Анализ формы
        return self.recognize_suit_by_shape(suit_roi), roi_vis_image
    
    def recognize_suit_by_color(self, corner):
        """
        Определение масти по цвету с использованием параметров
        """
        hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
        
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        red_pixels = cv2.countNonZero(red_mask)
        
        if red_pixels > self.params['color_red_threshold']:
            if self.is_heart_shape(corner):
                return 'hearts'
            else:
                return 'diamonds'
        else:
            if self.is_spade_shape(corner):
                return 'spades'
            else:
                return 'clubs'
    
    def is_heart_shape(self, image):
        """
        Определение формы сердца с использованием параметров
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = area / hull_area
                return solidity < self.params['shape_heart_solidity']
        
        return False
    
    def is_spade_shape(self, image):
        """
        Определение формы пики с использованием параметров
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        moments = cv2.moments(thresh)
        if moments['m00'] != 0:
            cy = moments['m01'] / moments['m00']
            height = image.shape[0]
            return cy < height * self.params['shape_spade_center']
        
        return False
    
    def recognize_suit_by_shape(self, corner):
        """
        Определение масти по форме
        """
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        
        if vertices < 8:
            if self.is_diamond_shape(contour):
                return 'diamonds'
            else:
                return 'clubs'
        else:
            if self.is_heart_shape(corner):
                return 'hearts'
            else:
                return 'spades'
    
    def is_diamond_shape(self, contour):
        """
        Проверка на форму ромба
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        box_area = cv2.contourArea(box)
        contour_area = cv2.contourArea(contour)
        
        if box_area > 0:
            return abs(box_area - contour_area) / box_area < 0.3
        return False
    
    def identify_card(self, image, contour, card_index=0):
        """
        Полная идентификация одной карты с визуализацией
        
        Returns:
            display_name: имя для отображения (с символами)
            rank: ранг
            suit: масть
            confidence: уверенность (0-100)
            vis_image: изображение с визуализацией
        """
        # Извлекаем угол карты с визуализацией
        corner, warped, vis_image = self.extract_card_corner(image, contour)
        
        if corner is None:
            return "Unknown", None, None, 0, image
        
        # Находим центр карты для визуализации
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            card_center = (cx, cy)
        else:
            card_center = None
        
        # Распознаем ранг и масть с визуализацией
        rank, vis_image = self.recognize_rank(corner, vis_image, card_index, card_center)
        suit, vis_image = self.recognize_suit(corner, vis_image, card_index, card_center)
        
        # Вычисляем уверенность
        confidence = 0
        if rank:
            confidence += 50
        if suit:
            confidence += 50
        
        # Формируем результат
        if rank and suit:
            display_name = f"{rank}{self.suit_unicode.get(suit, '?')}"
            return display_name, rank, suit, confidence, vis_image
        elif rank:
            return f"{rank}?", rank, None, confidence, vis_image
        elif suit:
            return f"?{self.suit_unicode.get(suit, '?')}", None, suit, confidence, vis_image
        else:
            return "Unknown", None, None, 0, vis_image
    
    def batch_identify(self, image, contours):
        """
        Идентификация нескольких карт с визуализацией
        
        Returns:
            list: список результатов для каждой карты
            vis_image: итоговое изображение с визуализацией
        """
        results = []
        vis_image = image.copy()
        
        for i, contour in enumerate(contours):
            name, rank, suit, conf, card_vis = self.identify_card(image, contour, i)
            
            # Обновляем визуализацию
            # Копируем только область вокруг карты, чтобы не затереть предыдущие
            x, y, w, h = cv2.boundingRect(contour)
            padding = 50
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            if card_vis is not None:
                vis_image[y_start:y_end, x_start:x_end] = card_vis[y_start:y_end, x_start:x_end]
            
            results.append({
                'index': i + 1,
                'contour': contour,
                'name': name,
                'rank': rank,
                'suit': suit,
                'confidence': conf
            })
        
        return results, vis_image