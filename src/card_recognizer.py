import cv2
import numpy as np
import pytesseract
import re
import os

# Настройка пути к Tesseract (для Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class CardRecognizer:
    def __init__(self, tesseract_path=None):
        """
        Инициализация распознавателя карт
        :param tesseract_path: путь к исполняемому файлу Tesseract (для Windows)
        """
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Словарь для преобразования OCR результатов в стандартные обозначения
        self.rank_mapping = {
            'A': 'A', 'ACE': 'A',
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
            'J': 'J', 'JACK': 'J',
            'Q': 'Q', 'QUEEN': 'Q',
            'K': 'K', 'KING': 'K'
        }
        
        # Словарь для мастей (Tesseract может распознавать символы)
        self.suit_mapping = {
            '♥': 'hearts', '❤': 'hearts', 'heart': 'hearts',
            '♦': 'diamonds', 'diamond': 'diamonds',
            '♣': 'clubs', 'club': 'clubs',
            '♠': 'spades', 'spade': 'spades'
        }
        
        # Настройки для предобработки изображения
        self.preprocess_config = {
            'threshold_block': 11,
            'threshold_c': 2,
            'gaussian_blur': 3,
            'morphology_kernel': 2
        }
    
    def extract_card_corner(self, image, contour):
        """
        Извлечение угла карты с мастью и достоинством
        """
        try:
            # Получаем ограничивающий прямоугольник с поворотом
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Определяем размеры для выпрямления
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Убеждаемся, что ширина - это большая сторона
            if height > width:
                width, height = height, width
                angle = rect[2] + 90
            else:
                angle = rect[2]
            
            # Получаем матрицу поворота
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], 
                              dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            # Вырезаем верхний левый угол (где обычно масть и достоинство)
            corner_size = min(width, height) // 4
            corner = warped[0:corner_size, 0:corner_size]
            
            # Добавляем отступы для лучшего распознавания
            corner_with_padding = cv2.copyMakeBorder(
                corner, 10, 10, 10, 10, 
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            
            return corner_with_padding, warped
            
        except Exception as e:
            print(f"Ошибка при извлечении угла: {e}")
            return None, None
    
    def preprocess_for_ocr(self, roi):
        """
        Предобработка изображения для улучшения OCR
        """
        if roi is None or roi.size == 0:
            return None
        
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Увеличиваем изображение для лучшего распознавания
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Применяем размытие для уменьшения шума
        blurred = cv2.GaussianBlur(scaled, 
                                   (self.preprocess_config['gaussian_blur'], 
                                    self.preprocess_config['gaussian_blur']), 0)
        
        # Применяем адаптивную пороговую обработку
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       self.preprocess_config['threshold_block'],
                                       self.preprocess_config['threshold_c'])
        
        # Морфологические операции для улучшения
        kernel = np.ones((self.preprocess_config['morphology_kernel'], 
                         self.preprocess_config['morphology_kernel']), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Инвертируем, если нужно (текст темный на светлом фоне)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        return thresh
    
    def recognize_rank(self, roi):
        """
        Распознавание достоинства карты (2-10, J, Q, K, A)
        """
        if roi is None:
            return None
        
        # Предобработка
        processed = self.preprocess_for_ocr(roi)
        if processed is None:
            return None
        
        # Сохраняем для отладки
        # cv2.imwrite('debug_rank.jpg', processed)
        
        # Настройки Tesseract для распознавания только цифр и букв
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789AJQK'
        
        try:
            # Пробуем распознать
            text = pytesseract.image_to_string(processed, config=custom_config)
            text = text.strip().upper()
            
            # Очищаем результат
            for pattern, replacement in self.rank_mapping.items():
                if pattern in text:
                    return replacement
            
            # Если ничего не нашли, пробуем другие настройки
            if not text:
                custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789AJQK'
                text = pytesseract.image_to_string(processed, config=custom_config)
                text = text.strip().upper()
                
                for pattern, replacement in self.rank_mapping.items():
                    if pattern in text:
                        return replacement
            
            # Если все еще ничего, пробуем с другими параметрами
            if not text:
                # Увеличиваем контраст
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(processed)
                
                text = pytesseract.image_to_string(enhanced, config=custom_config)
                text = text.strip().upper()
                
                for pattern, replacement in self.rank_mapping.items():
                    if pattern in text:
                        return replacement
            
            return None
            
        except Exception as e:
            print(f"Ошибка OCR при распознавании достоинства: {e}")
            return None
    
    def recognize_suit(self, roi):
        """
        Распознавание масти карты (♥, ♦, ♣, ♠)
        """
        if roi is None:
            return None
        
        # Предобработка
        processed = self.preprocess_for_ocr(roi)
        if processed is None:
            return None
        
        # Сохраняем для отладки
        cv2.imwrite('debug_suit.jpg', processed)
        
        # Для мастей используем более широкий набор символов
        custom_config = r'--oem 3 --psm 8'
        
        try:
            text = pytesseract.image_to_string(processed, config=custom_config)
            text = text.strip()
            
            # Проверяем соответствие мастям
            for symbol, suit in self.suit_mapping.items():
                if symbol in text:
                    return suit
            
            # Анализ цвета для определения масти
            # (красные ♥♦, черные ♣♠)
            if self.is_red_suit(roi):
                # Если красная, то либо ♥ либо ♦
                # Анализируем форму
                if self.is_heart_shape(roi):
                    return 'hearts'
                else:
                    return 'diamonds'
            else:
                # Если черная, то либо ♣ либо ♠
                if self.is_spade_shape(roi):
                    return 'spades'
                else:
                    return 'clubs'
            
        except Exception as e:
            print(f"Ошибка OCR при распознавании масти: {e}")
            return None
    
    def is_red_suit(self, roi):
        """
        Определение, является ли масть красной
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Красный цвет в HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        return red_pixels > total_pixels * 0.1  # 10% красных пикселей
    
    def is_heart_shape(self, roi):
        """
        Простой анализ формы для определения ♥
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Сердце обычно имеет большую выпуклость
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            
            if hull_area > 0:
                solidity = contour_area / hull_area
                # Сердце имеет меньшую заполненность из-за выемки
                return solidity < 0.85
        
        return False
    
    def is_spade_shape(self, roi):
        """
        Простой анализ формы для определения ♠
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Пика обычно имеет высокий центр тяжести
        moments = cv2.moments(thresh)
        if moments['m00'] != 0:
            cy = moments['m01'] / moments['m00']
            height = roi.shape[0]
            # Центр тяжести выше середины
            return cy < height * 0.4
        
        return False
    
    def segment_corner(self, corner):
        """
        Сегментация угла на область достоинства и масти
        """
        if corner is None:
            return None, None
        
        height, width = corner.shape[:2]
        
        # Предполагаем, что достоинство в верхней части, масть в нижней
        rank_roi = corner[0:height//2, :]
        suit_roi = corner[height//2:height, :]
        
        return rank_roi, suit_roi
    
    def identify_card(self, image, contour):
        """
        Полная идентификация карты
        """
        # Извлекаем угол карты
        corner, warped_card = self.extract_card_corner(image, contour)
        
        if corner is None:
            return "Unknown", None, None
        
        # Сегментируем на достоинство и масть
        rank_roi, suit_roi = self.segment_corner(corner)
        
        # Распознаем достоинство
        rank = self.recognize_rank(rank_roi)
        
        # Распознаем масть
        suit = self.recognize_suit(suit_roi)
        
        if rank and suit:
            card_name = f"{rank}{suit}"
            # Конвертируем в символы мастей для отображения
            suit_symbol = self.get_suit_symbol(suit)
            display_name = f"{rank}{suit_symbol}"
            return display_name, rank, suit
        elif rank:
            return f"{rank}?", rank, None
        elif suit:
            return f"?{suit}", None, suit
        else:
            return "Unknown", None, None
    
    def get_suit_symbol(self, suit):
        """
        Получение символа масти для отображения
        """
        symbols = {
            'hearts': '♥',
            'diamonds': '♦',
            'clubs': '♣',
            'spades': '♠'
        }
        return symbols.get(suit, '?')

    def batch_recognize_cards(self, image, contours):
        """
        Распознавание нескольких карт на изображении
        """
        results = []
        for i, contour in enumerate(contours):
            card_name, rank, suit = self.identify_card(image, contour)
            results.append({
                'index': i + 1,
                'contour': contour,
                'name': card_name,
                'rank': rank,
                'suit': suit
            })
        return results