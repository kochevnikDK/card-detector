import cv2
import numpy as np
import os
import glob

class TemplateMatcher:
    """
    Класс для распознавания карт с помощью шаблонов
    Используется как дополнение к OCR для повышения точности
    """
    def __init__(self, templates_path="templates"):
        """
        Инициализация матчера шаблонов
        
        Args:
            templates_path: путь к папке с шаблонами
        """
        self.templates_path = templates_path
        self.rank_templates = {}
        self.suit_templates = {}
        self.load_templates()
    
    def load_templates(self):
        """Загрузка шаблонов из папки templates"""
        # Загрузка шаблонов рангов
        rank_path = os.path.join(self.templates_path, "ranks")
        if os.path.exists(rank_path):
            for template_file in glob.glob(os.path.join(rank_path, "*.png")):
                rank_name = os.path.splitext(os.path.basename(template_file))[0]
                template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.rank_templates[rank_name] = template
        
        # Загрузка шаблонов мастей
        suit_path = os.path.join(self.templates_path, "suits")
        if os.path.exists(suit_path):
            for template_file in glob.glob(os.path.join(suit_path, "*.png")):
                suit_name = os.path.splitext(os.path.basename(template_file))[0]
                template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.suit_templates[suit_name] = template
        
        print(f"Загружено шаблонов: {len(self.rank_templates)} рангов, {len(self.suit_templates)} мастей")
    
    def match_rank(self, roi, threshold=0.7):
        """
        Сопоставление ранга с шаблонами
        
        Args:
            roi: область интереса (изображение ранга)
            threshold: порог уверенности
            
        Returns:
            best_rank: лучший найденный ранг
            best_score: лучшая оценка схожести
        """
        if roi is None or roi.size == 0:
            return None, 0
        
        # Конвертируем в grayscale если нужно
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Применяем пороговую обработку
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        best_rank = None
        best_score = 0
        
        for rank_name, template in self.rank_templates.items():
            # Пробуем разные масштабы
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                if scale <= 0:
                    continue
                    
                # Изменяем размер шаблона
                h, w = template.shape
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w > thresh.shape[1] or new_h > thresh.shape[0]:
                    continue
                
                resized_template = cv2.resize(template, (new_w, new_h))
                
                # Сопоставление шаблона
                result = cv2.matchTemplate(thresh, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score and max_val > threshold:
                    best_score = max_val
                    best_rank = rank_name
        
        return best_rank, best_score
    
    def match_suit(self, roi, threshold=0.7):
        """
        Сопоставление масти с шаблонами
        
        Args:
            roi: область интереса (изображение масти)
            threshold: порог уверенности
            
        Returns:
            best_suit: лучшая найденная масть
            best_score: лучшая оценка схожести
        """
        if roi is None or roi.size == 0:
            return None, 0
        
        # Конвертируем в grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Применяем пороговую обработку
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        best_suit = None
        best_score = 0
        
        for suit_name, template in self.suit_templates.items():
            # Пробуем разные масштабы
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                if scale <= 0:
                    continue
                    
                # Изменяем размер шаблона
                h, w = template.shape
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w > thresh.shape[1] or new_h > thresh.shape[0]:
                    continue
                
                resized_template = cv2.resize(template, (new_w, new_h))
                
                # Сопоставление шаблона
                result = cv2.matchTemplate(thresh, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score and max_val > threshold:
                    best_score = max_val
                    best_suit = suit_name
        
        return best_suit, best_score
    
    def create_templates_from_image(self, image, contours, output_path="templates"):
        """
        Создание шаблонов из распознанных карт
        Полезно для обучения на своих изображениях
        """
        os.makedirs(os.path.join(output_path, "ranks"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "suits"), exist_ok=True)
        
        for i, contour in enumerate(contours):
            # Извлекаем угол карты
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Выпрямляем карту
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            if width < height:
                width, height = height, width
            
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], 
                              dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            # Вырезаем угол
            corner_size = min(width, height) // 4
            corner = warped[10:corner_size+10, 10:corner_size+10]
            
            # Разделяем на ранг и масть
            h, w = corner.shape[:2]
            rank_roi = corner[0:h//2, 0:w]
            suit_roi = corner[h//2:h, 0:w]
            
            # Сохраняем как шаблоны
            cv2.imwrite(os.path.join(output_path, "ranks", f"rank_{i+1}.png"), rank_roi)
            cv2.imwrite(os.path.join(output_path, "suits", f"suit_{i+1}.png"), suit_roi)
        
        print(f"Создано {len(contours)} шаблонов в папке {output_path}")