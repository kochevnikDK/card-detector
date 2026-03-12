import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch

class YOLOCardDetector:
    """
    Детектор игральных карт на основе YOLOv8
    Обнаруживает и классифицирует карты за один проход
    """
    
    def __init__(self, model_path=None, conf_threshold=0.5, device=None):
        """
        Инициализация YOLO детектора
        
        Args:
            model_path: путь к .pt файлу модели (если None, использует yolov8n.pt)
            conf_threshold: порог уверенности (0-1)
            device: 'cpu', 'cuda', или None (автоопределение)
        """
        # Определяем устройство
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Используется устройство: {self.device}")
        
        # Загружаем модель
        if model_path and os.path.exists(model_path):
            print(f"Загрузка модели из {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Загрузка предобученной модели YOLOv8n")
            self.model = YOLO('yolov8n.pt')  # Базовая модель
            
        self.conf_threshold = conf_threshold
        
        # Маппинг классов (будет обновлен при загрузке кастомной модели)
        self.class_names = {}
        
        # Параметры для визуализации
        self.params = {
            'show_boxes': True,
            'show_labels': True,
            'box_thickness': 2,
            'conf_color': (0, 255, 0),
            'text_color': (255, 255, 255),
            'text_bg_color': (0, 0, 0)
        }
        
    def set_params(self, params):
        """Обновление параметров"""
        self.params.update(params)
        if 'conf_threshold' in params:
            self.conf_threshold = params['conf_threshold']
    
    def detect_cards(self, image_path, return_visualization=True):
        """
        Детектирование карт на изображении
        
        Args:
            image_path: путь к изображению
            return_visualization: возвращать ли визуализацию
            
        Returns:
            results: список обнаруженных карт
            vis_image: изображение с визуализацией (если return_visualization=True)
        """
        # Загружаем изображение
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        # Выполняем инференс
        results = self.model(img, conf=self.conf_threshold, device=self.device)
        
        # Получаем результаты для первого изображения
        result = results[0]
        
        # Формируем список обнаруженных карт
        cards = []
        vis_image = img.copy() if return_visualization else None
        
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                # Координаты bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Уверенность
                conf = float(box.conf[0].cpu().numpy())
                
                # Класс
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Создаем контур для совместимости с существующим кодом
                contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                
                card_info = {
                    'index': i + 1,
                    'bbox': (x1, y1, x2, y2),
                    'contour': contour,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': int(conf * 100),
                    'name': class_name,  # Для совместимости
                    'rank': self._extract_rank(class_name),
                    'suit': self._extract_suit(class_name)
                }
                cards.append(card_info)
                
                # Визуализация
                if return_visualization:
                    # Рисуем bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), 
                                 self.params['conf_color'], self.params['box_thickness'])
                    
                    # Подготовка текста
                    label = f"{class_name} ({conf:.2f})"
                    
                    # Рисуем фон для текста
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_image, (x1, y1 - text_h - 10), 
                                (x1 + text_w + 10, y1), self.params['text_bg_color'], -1)
                    
                    # Рисуем текст
                    cv2.putText(vis_image, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.params['text_color'], 2)
        
        if return_visualization:
            return cards, vis_image
        return cards
    
    def _extract_rank(self, class_name):
        """Извлечение ранга из имени класса (для карт)"""
        # Ожидаем формат типа "10h", "As", "Kd" и т.д.
        if len(class_name) >= 2:
            rank_part = class_name[:-1]  # Все кроме последнего символа
            if rank_part in ['10', 'J', 'Q', 'K', 'A'] or rank_part.isdigit():
                return rank_part
        return None
    
    def _extract_suit(self, class_name):
        """Извлечение масти из имени класса"""
        suit_map = {
            'h': 'hearts',
            'd': 'diamonds', 
            'c': 'clubs',
            's': 'spades'
        }
        if len(class_name) >= 1:
            suit_code = class_name[-1].lower()
            return suit_map.get(suit_code)
        return None
    
    def detect_from_webcam(self, callback=None):
        """
        Детектирование в реальном времени с веб-камеры
        """
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Инференс
            results = self.model(frame, conf=self.conf_threshold, device=self.device)
            
            # Визуализация
            annotated_frame = results[0].plot()
            
            # Отображаем
            cv2.imshow('YOLO Card Detection', annotated_frame)
            
            if callback:
                callback(results[0])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def get_recognition_results(self):
        """Для совместимости с существующим кодом"""
        return []  # Будет обновляться при вызове detect_cards