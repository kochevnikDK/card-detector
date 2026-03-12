import cv2
import numpy as np
import threading
import time
import os
from src.yolo_detector import YOLOCardDetector
from src.ocr_recognizer import OCRCardRecognizer

class CardDetector:
    def __init__(self, method='yolo', tesseract_path=None, model_path=None):
        """
        Универсальный детектор с выбором метода
        
        Args:
            method: 'yolo', 'ocr', или 'hybrid'
            tesseract_path: путь к tesseract.exe (для OCR метода)
            model_path: путь к YOLO модели
        """
        self.method = method
        self.params = self._get_default_params()
        
        # Инициализация выбранного метода
        if method in ['yolo', 'hybrid']:
            print(f"Инициализация YOLO детектора... (метод: {method})")
            try:
                self.yolo_detector = YOLOCardDetector(
                    model_path=model_path,
                    conf_threshold=self.params.get('conf_threshold', 0.5)
                )
                print("✓ YOLO детектор успешно инициализирован")
            except Exception as e:
                print(f"✗ Ошибка инициализации YOLO: {e}")
                self.yolo_detector = None
        else:
            self.yolo_detector = None
            
        if method in ['ocr', 'hybrid']:
            print(f"Инициализация OCR распознавателя... (метод: {method})")
            try:
                self.ocr_recognizer = OCRCardRecognizer(tesseract_path)
                print("✓ OCR распознаватель успешно инициализирован")
            except Exception as e:
                print(f"✗ Ошибка инициализации OCR: {e}")
                self.ocr_recognizer = None
        else:
            self.ocr_recognizer = None
        
        # Общие параметры
        self.last_contours = []
        self.current_image = None
        self.current_image_path = None
        self.last_recognition_results = []
        self.last_result_image = None
        self.last_num_cards = 0
        self.last_stages = {}
        
        # Для оптимизации
        self.processing = False
        self.pending_update = False
        self.last_process_time = 0
        self.min_process_interval = 0.1
        self.thread_lock = threading.Lock()
        
        # Кэш результатов
        self.result_cache = {}
        self.cache_size = 5
        
        print(f"✓ CardDetector инициализирован с методом: {method}")
    
    def _get_default_params(self):
        """Параметры по умолчанию"""
        return {
            # Общие параметры
            'method': 'yolo',
            'use_recognition': True,
            
            # Параметры YOLO
            'conf_threshold': 0.5,
            'show_boxes': True,
            'show_labels': True,
            'box_thickness': 2,
            'conf_color': (0, 255, 0),
            'text_color': (255, 255, 255),
            'text_bg_color': (0, 0, 0),
            
            # Параметры OCR (для совместимости)
            'gaussian_blur': 5,
            'threshold_block': 11,
            'threshold_c': 2,
            'canny_threshold1': 50,
            'canny_threshold2': 150,
            'min_card_area': 5000,
            'max_card_area': 500000,
            'approx_epsilon': 0.02,
            'use_morphology': True,
            'morphology_kernel': 5,
            'use_color_filter': True,
            'use_edge_enhancement': True,
            'edge_enhancement_strength': 1.5,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.5,
            'min_solidity': 0.7,
            'contour_approx_epsilon': 0.02,
            
            # Параметры распознавания (для OCR)
            'corner_padding': 10,
            'corner_size_factor': 4,
            'corner_zoom_factor': 2,
            'rank_vertical_split': 0.5,
            'preprocess_threshold_block': 11,
            'preprocess_threshold_c': 2,
            'preprocess_gaussian_blur': 3,
            'ocr_rank_psm': 8,
            'ocr_suit_psm': 8,
            'show_corner': True,
            'show_roi': True,
            'show_text': True,
            'roi_line_thickness': 2,
            'roi_color_rank': (255, 0, 0),
            'roi_color_suit': (0, 255, 0),
            'roi_color_corner': (0, 255, 255)
        }
    
    def set_params(self, params):
        """Обновление параметров детектора"""
        with self.thread_lock:
            self.params.update(params)
            
            # Обновляем параметры YOLO
            if self.yolo_detector:
                yolo_params = {}
                for key in ['conf_threshold', 'show_boxes', 'show_labels', 
                           'box_thickness', 'conf_color', 'text_color', 'text_bg_color']:
                    if key in params:
                        yolo_params[key] = params[key]
                if yolo_params:
                    self.yolo_detector.set_params(yolo_params)
            
            # Обновляем параметры OCR
            if self.ocr_recognizer:
                ocr_params = {}
                for key in ['corner_padding', 'corner_size_factor', 'corner_zoom_factor',
                           'rank_vertical_split', 'preprocess_threshold_block',
                           'preprocess_threshold_c', 'preprocess_gaussian_blur',
                           'ocr_rank_psm', 'ocr_suit_psm', 'show_corner', 'show_roi',
                           'show_text', 'roi_line_thickness']:
                    if key in params:
                        ocr_params[key] = params[key]
                if ocr_params:
                    self.ocr_recognizer.set_params(ocr_params)
            
            self.pending_update = True
    
    def _get_cache_key(self, image_path, params):
        """Создание ключа кэша"""
        # Используем путь к файлу и основные параметры
        key_params = (
            params.get('method', 'yolo'),
            params.get('conf_threshold', 0.5),
            params.get('gaussian_blur', 5),
            params.get('threshold_block', 11),
        )
        return f"{image_path}_{key_params}"
    
    def preprocess_image(self, img):
        """
        Предобработка изображения (для совместимости с OCR)
        Упрощенная версия для YOLO
        """
        stages = {}
        
        # Базовая предобработка
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stages['gray'] = gray
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        stages['hsv'] = hsv
        
        # Улучшение контраста
        if self.params.get('use_edge_enhancement', False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            stages['enhanced'] = enhanced
        else:
            enhanced = gray
        
        # Размытие
        kernel_size = self.params.get('gaussian_blur', 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)
        stages['blurred'] = blurred
        
        return blurred, stages
    
    def process_image_internal(self):
        """Внутренний метод обработки изображения"""
        if self.current_image_path is None:
            return None, 0, {}, []
        
        # Проверяем кэш
        cache_key = self._get_cache_key(self.current_image_path, self.params)
        if cache_key in self.result_cache:
            print("✓ Используем кэшированный результат")
            cached = self.result_cache[cache_key]
            return cached['image'], cached['num_cards'], cached['stages'], cached['results']
        
        # Загружаем изображение
        img = cv2.imread(self.current_image_path)
        if img is None:
            print(f"✗ Ошибка загрузки изображения: {self.current_image_path}")
            return None, 0, {}, []
        
        self.current_image = img.copy()
        
        # Изменение размера для ускорения (если нужно)
        height, width = img.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            self.current_image = cv2.resize(self.current_image, (new_width, new_height))
        
        stages = {}
        recognition_results = []
        result_img = img.copy()
        
        # Выбираем метод детекции
        try:
            if self.method == 'yolo' and self.yolo_detector:
                # YOLO метод
                print("✓ Обработка через YOLO...")
                cards, vis_image = self.yolo_detector.detect_cards(
                    self.current_image_path, return_visualization=True
                )
                
                # Конвертируем результаты
                for card in cards:
                    recognition_results.append({
                        'index': card['index'],
                        'contour': card['contour'],
                        'bbox': card['bbox'],
                        'name': card['name'],
                        'rank': card['rank'],
                        'suit': card['suit'],
                        'confidence': card['confidence']
                    })
                
                self.last_contours = [card['contour'] for card in cards]
                result_img = vis_image
                
            elif self.method == 'ocr' and self.ocr_recognizer:
                # OCR метод
                print("✓ Обработка через OCR...")
                # Здесь должен быть ваш существующий код OCR
                # Для краткости оставляем заглушку
                recognition_results = []
                result_img = img
                
            elif self.method == 'hybrid' and self.yolo_detector and self.ocr_recognizer:
                # Гибридный метод
                print("✓ Обработка через гибридный метод...")
                cards, vis_image = self.yolo_detector.detect_cards(
                    self.current_image_path, return_visualization=True
                )
                
                # Верификация через OCR при низкой уверенности
                for card in cards:
                    if card['confidence'] < 70 and self.ocr_recognizer:
                        # OCR верификация
                        ocr_name, rank, suit, conf, _ = self.ocr_recognizer.identify_card(
                            self.current_image, card['contour'], card['index']
                        )
                        if conf > card['confidence']:
                            card['name'] = ocr_name
                            card['rank'] = rank
                            card['suit'] = suit
                            card['confidence'] = conf
                    
                    recognition_results.append(card)
                
                self.last_contours = [card['contour'] for card in cards]
                result_img = vis_image
            
            else:
                print(f"✗ Метод {self.method} не поддерживается или не инициализирован")
                return None, 0, {}, []
            
            # Сохраняем в кэш
            if len(self.result_cache) >= self.cache_size:
                # Удаляем самый старый
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            
            self.result_cache[cache_key] = {
                'image': result_img,
                'num_cards': len(recognition_results),
                'stages': stages,
                'results': recognition_results
            }
            
            print(f"✓ Обработка завершена. Найдено карт: {len(recognition_results)}")
            return result_img, len(recognition_results), stages, recognition_results
            
        except Exception as e:
            print(f"✗ Ошибка при обработке: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, {}, []
    
    def process_image_async(self, callback=None):
        """Асинхронная обработка изображения"""
        def process():
            with self.thread_lock:
                if self.processing:
                    print("⚠ Предыдущая обработка еще выполняется")
                    return
                self.processing = True
                self.pending_update = False
            
            try:
                # Устанавливаем таймаут 30 секунд
                import signal
                
                class TimeoutError(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Превышено время обработки")
                
                # Таймаут работает только в Linux/Mac
                if os.name != 'nt':  # Не для Windows
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)
                
                try:
                    result_img, num_cards, stages, results = self.process_image_internal()
                except TimeoutError as e:
                    print(f"✗ {e}")
                    result_img, num_cards, stages, results = None, 0, {}, []
                finally:
                    if os.name != 'nt':
                        signal.alarm(0)
                
                with self.thread_lock:
                    self.last_result_image = result_img
                    self.last_num_cards = num_cards
                    self.last_stages = stages
                    self.last_recognition_results = results
                    self.last_process_time = time.time()
                    self.processing = False
                
                if callback and result_img is not None:
                    callback(result_img, num_cards, stages, results)
                    
            except Exception as e:
                print(f"✗ Ошибка в асинхронной обработке: {e}")
                with self.thread_lock:
                    self.processing = False
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def detect_cards_advanced(self, image_path, return_all=False, async_mode=False, callback=None):
        """
        Усовершенствованный метод детектирования карт
        """
        self.current_image_path = image_path
        
        if async_mode:
            self.process_image_async(callback)
            if return_all:
                return None, 0, {}
            return None, 0
        else:
            result_img, num_cards, stages, results = self.process_image_internal()
            with self.thread_lock:
                self.last_result_image = result_img
                self.last_num_cards = num_cards
                self.last_stages = stages
                self.last_recognition_results = results
            
            if return_all:
                return result_img, num_cards, stages
            return result_img, num_cards
    
    def detect_cards(self, image_path, return_all=False, async_mode=False, callback=None):
        """Основной метод детектирования"""
        return self.detect_cards_advanced(image_path, return_all, async_mode, callback)
    
    def get_recognition_results(self):
        """Получение результатов распознавания"""
        with self.thread_lock:
            if self.last_recognition_results:
                return self.last_recognition_results.copy()
        return []
    
    def get_last_result(self):
        """Получение последнего результата обработки"""
        with self.thread_lock:
            return (self.last_result_image.copy() if self.last_result_image is not None else None,
                   self.last_num_cards,
                   self.last_stages.copy() if self.last_stages else {})
    
    def is_processing(self):
        """Проверка, идет ли обработка"""
        with self.thread_lock:
            return self.processing
    
    def check_for_updates(self):
        """Проверка наличия ожидающих обновлений"""
        with self.thread_lock:
            if self.pending_update and not self.processing:
                self.pending_update = False
                return True
        return False
    
    def clear_cache(self):
        """Очистка кэша"""
        with self.thread_lock:
            self.result_cache.clear()
            print("✓ Кэш очищен")