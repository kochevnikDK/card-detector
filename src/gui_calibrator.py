import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import sys
import time

# Добавляем путь к родительской папке
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.card_detector import CardDetector

class CardDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Калибратор детектора игральных карт")
        self.root.geometry("1500x900")
        
        # Инициализация детектора (по умолчанию YOLO)
        self.detector = None
        self.current_image = None
        self.current_image_path = None
        self.result_image = None
        self.stages = None
        
        # Для оптимизации
        self.update_timer = None
        self.update_delay = 300  # Задержка обновления в миллисекундах
        self.last_update_time = 0
        
        # Создание интерфейса
        self.create_widgets()
        
        # Привязка обработчиков событий
        self.setup_bindings()
        
        # Инициализация детектора после создания GUI
        self.init_detector()
        
        # Запуск периодической проверки обновлений
        self.check_updates()
    
    def init_detector(self):
        """Инициализация детектора с текущими параметрами"""
        try:
            method = self.method_var.get()
            model_path = self.model_path_var.get().strip()
            tesseract_path = self.tesseract_path_var.get().strip()
            
            # Проверяем существование файла модели
            if method in ['yolo', 'hybrid'] and model_path and not os.path.exists(model_path):
                print(f"⚠ Модель не найдена: {model_path}")
                messagebox.showwarning("Предупреждение", 
                    f"Файл модели не найден:\n{model_path}\n\nБудет использована базовая модель YOLO.")
                model_path = None
            
            self.detector = CardDetector(
                method=method,
                model_path=model_path if model_path else None,
                tesseract_path=tesseract_path if tesseract_path else None
            )
            
            # Обновляем параметры
            self.update_params()
            
            print(f"✓ Детектор инициализирован с методом: {method}")
            
        except Exception as e:
            print(f"✗ Ошибка инициализации детектора: {e}")
            messagebox.showerror("Ошибка", f"Не удалось инициализировать детектор:\n{str(e)}")
    
    def create_widgets(self):
        # Основной контейнер с изменяемыми размерами
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Левая панель с параметрами
        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=1)
        
        # Создаем Canvas с прокруткой для левой панели
        canvas = tk.Canvas(left_frame, highlightthickness=0, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        
        # Фрейм для содержимого внутри Canvas
        self.scrollable_frame = ttk.Frame(canvas)
        
        # Настройка прокрутки
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=430)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Размещаем Canvas и Scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Привязываем колесико мыши к прокрутке
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # ===== Параметры =====
        
        # Выбор метода детектирования
        method_frame = ttk.LabelFrame(self.scrollable_frame, text="Метод детектирования", padding="5")
        method_frame.pack(fill="x", padx=5, pady=5)
        
        self.method_var = tk.StringVar(value="yolo")
        methods = [
            ("YOLO (быстрый, точный)", "yolo"),
            ("OCR (Tesseract)", "ocr"),
            ("Гибридный (YOLO + OCR)", "hybrid")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, 
                           value=value, command=self.on_method_change).pack(anchor="w", pady=2)
        
        # ===== Параметры YOLO =====
        self.yolo_frame = ttk.LabelFrame(self.scrollable_frame, text="Параметры YOLO", padding="5")
        self.yolo_frame.pack(fill="x", padx=5, pady=5)
        
        # Путь к модели
        row_y1 = ttk.Frame(self.yolo_frame)
        row_y1.pack(fill="x", pady=2)
        ttk.Label(row_y1, text="Путь к модели:", width=15).pack(side="left")
        self.model_path_var = tk.StringVar(value="models/card_model/best.pt")
        ttk.Entry(row_y1, textvariable=self.model_path_var, width=30).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(row_y1, text="Обзор", command=self.browse_model).pack(side="left")
        
        # Порог уверенности
        row_y2 = ttk.Frame(self.yolo_frame)
        row_y2.pack(fill="x", pady=2)
        ttk.Label(row_y2, text="Порог уверенности:", width=15).pack(side="left")
        self.conf_threshold_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(row_y2, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                               variable=self.conf_threshold_var, command=self.on_param_change_delayed)
        conf_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.conf_label = ttk.Label(row_y2, text="0.5", width=3)
        self.conf_label.pack(side="left")
        
        # Параметры визуализации YOLO
        row_y3 = ttk.Frame(self.yolo_frame)
        row_y3.pack(fill="x", pady=2)
        self.show_boxes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_y3, text="Показывать рамки", 
                       variable=self.show_boxes_var, command=self.on_param_change_delayed).pack(side="left", padx=5)
        
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_y3, text="Показывать подписи", 
                       variable=self.show_labels_var, command=self.on_param_change_delayed).pack(side="left", padx=5)
        
        # Толщина линий
        row_y4 = ttk.Frame(self.yolo_frame)
        row_y4.pack(fill="x", pady=2)
        ttk.Label(row_y4, text="Толщина линий:", width=15).pack(side="left")
        self.box_thickness_var = tk.IntVar(value=2)
        thickness_scale = ttk.Scale(row_y4, from_=1, to=5, orient=tk.HORIZONTAL,
                                    variable=self.box_thickness_var, command=self.on_param_change_delayed)
        thickness_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.thickness_label = ttk.Label(row_y4, text="2", width=3)
        self.thickness_label.pack(side="left")
        
        # ===== Параметры OCR =====
        self.ocr_frame = ttk.LabelFrame(self.scrollable_frame, text="Параметры OCR", padding="5")
        
        # Путь к Tesseract
        row_o1 = ttk.Frame(self.ocr_frame)
        row_o1.pack(fill="x", pady=2)
        ttk.Label(row_o1, text="Tesseract path:", width=15).pack(side="left")
        self.tesseract_path_var = tk.StringVar(value=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        ttk.Entry(row_o1, textvariable=self.tesseract_path_var, width=30).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(row_o1, text="Обзор", command=self.browse_tesseract).pack(side="left")
        
        # Параметры выделения угла
        corner_frame = ttk.LabelFrame(self.ocr_frame, text="Выделение угла", padding="5")
        corner_frame.pack(fill="x", pady=5)
        
        # Отступ
        row_c1 = ttk.Frame(corner_frame)
        row_c1.pack(fill="x", pady=2)
        ttk.Label(row_c1, text="Отступ угла:", width=15).pack(side="left")
        self.corner_padding_var = tk.IntVar(value=10)
        corner_padding_scale = ttk.Scale(row_c1, from_=0, to=30, orient=tk.HORIZONTAL,
                                        variable=self.corner_padding_var, command=self.on_param_change_delayed)
        corner_padding_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.corner_padding_label = ttk.Label(row_c1, text="10", width=3)
        self.corner_padding_label.pack(side="left")
        
        # Размер угла
        row_c2 = ttk.Frame(corner_frame)
        row_c2.pack(fill="x", pady=2)
        ttk.Label(row_c2, text="Размер угла (1/X):", width=15).pack(side="left")
        self.corner_size_var = tk.IntVar(value=4)
        corner_size_scale = ttk.Scale(row_c2, from_=2, to=10, orient=tk.HORIZONTAL,
                                    variable=self.corner_size_var, command=self.on_param_change_delayed)
        corner_size_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.corner_size_label = ttk.Label(row_c2, text="4", width=3)
        self.corner_size_label.pack(side="left")
        
        # Сегментация
        row_s1 = ttk.Frame(self.ocr_frame)
        row_s1.pack(fill="x", pady=2)
        ttk.Label(row_s1, text="Граница ранга:", width=15).pack(side="left")
        self.rank_split_var = tk.DoubleVar(value=0.5)
        rank_split_scale = ttk.Scale(row_s1, from_=0.3, to=0.7, orient=tk.HORIZONTAL,
                                    variable=self.rank_split_var, command=self.on_param_change_delayed)
        rank_split_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.rank_split_label = ttk.Label(row_s1, text="0.5", width=3)
        self.rank_split_label.pack(side="left")
        
        # ===== Общие параметры визуализации =====
        vis_frame = ttk.LabelFrame(self.scrollable_frame, text="Общие параметры визуализации", padding="5")
        vis_frame.pack(fill="x", padx=5, pady=5)
        
        self.show_corner_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Показывать угол карты", 
                       variable=self.show_corner_var, command=self.on_param_change_delayed).pack(anchor="w", pady=2)
        
        self.show_roi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Показывать ROI ранга/масти", 
                       variable=self.show_roi_var, command=self.on_param_change_delayed).pack(anchor="w", pady=2)
        
        self.show_text_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Показывать подписи", 
                       variable=self.show_text_var, command=self.on_param_change_delayed).pack(anchor="w", pady=2)
        
        # ===== Параметры формы (для совместимости) =====
        self.shape_params_frame = ttk.LabelFrame(self.scrollable_frame, text="Параметры формы", padding="5")
        self.shape_params_frame.pack(fill="x", padx=5, pady=5)
        
        # Соотношение сторон мин
        row1 = ttk.Frame(self.shape_params_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Мин. соотношение:", width=15).pack(side="left")
        self.min_aspect_var = tk.DoubleVar(value=0.5)
        min_aspect_scale = ttk.Scale(row1, from_=0.2, to=1.0, orient=tk.HORIZONTAL,
                                     variable=self.min_aspect_var, command=self.on_param_change_delayed)
        min_aspect_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.min_aspect_label = ttk.Label(row1, text="0.5", width=4)
        self.min_aspect_label.pack(side="left")
        
        # Соотношение сторон макс
        row2 = ttk.Frame(self.shape_params_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Макс. соотношение:", width=15).pack(side="left")
        self.max_aspect_var = tk.DoubleVar(value=2.5)
        max_aspect_scale = ttk.Scale(row2, from_=1.0, to=5.0, orient=tk.HORIZONTAL,
                                     variable=self.max_aspect_var, command=self.on_param_change_delayed)
        max_aspect_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.max_aspect_label = ttk.Label(row2, text="2.5", width=4)
        self.max_aspect_label.pack(side="left")
        
        # Кнопки управления
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        ttk.Button(button_frame, text="Загрузить изображение",
                  command=self.load_image).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Сохранить параметры",
                  command=self.save_params).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Загрузить параметры",
                  command=self.load_params).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Сохранить результат",
                  command=self.save_result).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Применить метод",
                  command=self.on_method_change).pack(fill="x", pady=2)
        
        # Информация о количестве карт
        self.info_label = ttk.Label(self.scrollable_frame, text="Карт обнаружено: 0",
                                   font=('Arial', 12, 'bold'), foreground="blue")
        self.info_label.pack(pady=10)
        
        # Индикатор обработки
        self.progress_label = ttk.Label(self.scrollable_frame, text="Готов",
                                       font=('Arial', 10), foreground="green")
        self.progress_label.pack(pady=5)
        
        # Правая панель с изображениями
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # Верхняя строка с изображениями
        top_row = ttk.Frame(right_frame)
        top_row.pack(fill="both", expand=True)
        
        # Оригинальное изображение
        original_frame = ttk.LabelFrame(top_row, text="Оригинал", padding="5")
        original_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg='gray', highlightthickness=0)
        self.original_canvas.pack(fill="both", expand=True)
        
        # Результат
        result_frame = ttk.LabelFrame(top_row, text="Результат", padding="5")
        result_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.result_canvas = tk.Canvas(result_frame, bg='gray', highlightthickness=0)
        self.result_canvas.pack(fill="both", expand=True)
        
        # Нижняя панель с информацией
        bottom_frame = ttk.Frame(right_frame)
        bottom_frame.pack(fill="x", padx=5, pady=5)
        
        # Текстовое поле для вывода результатов
        self.results_text = tk.Text(bottom_frame, height=8, width=50)
        self.results_text.pack(fill="both", expand=True)
        
        # Скроллбар для текстового поля
        scrollbar_text = ttk.Scrollbar(bottom_frame, orient="vertical", command=self.results_text.yview)
        scrollbar_text.pack(side="right", fill="y")
        self.results_text.configure(yscrollcommand=scrollbar_text.set)
    
    def browse_model(self):
        """Выбор файла модели"""
        filename = filedialog.askopenfilename(
            title="Выберите файл модели YOLO",
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
            self.on_method_change()
    
    def browse_tesseract(self):
        """Выбор пути к Tesseract"""
        filename = filedialog.askopenfilename(
            title="Выберите tesseract.exe",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")]
        )
        if filename:
            self.tesseract_path_var.set(filename)
            self.on_method_change()
    
    def setup_bindings(self):
        """Настройка дополнительных привязок событий"""
        variables = [
            self.conf_threshold_var, self.box_thickness_var,
            self.corner_padding_var, self.corner_size_var,
            self.rank_split_var, self.min_aspect_var,
            self.max_aspect_var
        ]
        
        for var in variables:
            var.trace('w', self.update_labels)
        
        # Для boolean переменных
        bool_vars = [
            self.show_boxes_var, self.show_labels_var,
            self.show_corner_var, self.show_roi_var, self.show_text_var
        ]
        
        for var in bool_vars:
            var.trace('w', self.on_param_change_delayed)
    
    def update_labels(self, *args):
        """Обновление текстовых меток для слайдеров"""
        self.conf_label.config(text=f"{self.conf_threshold_var.get():.2f}")
        self.thickness_label.config(text=str(self.box_thickness_var.get()))
        self.corner_padding_label.config(text=str(self.corner_padding_var.get()))
        self.corner_size_label.config(text=str(self.corner_size_var.get()))
        self.rank_split_label.config(text=f"{self.rank_split_var.get():.2f}")
        self.min_aspect_label.config(text=f"{self.min_aspect_var.get():.2f}")
        self.max_aspect_label.config(text=f"{self.max_aspect_var.get():.2f}")
    
    def on_method_change(self, *args):
        """Обработчик изменения метода детектирования"""
        # Переинициализируем детектор
        self.init_detector()
        
        # Показываем/скрываем соответствующие фреймы
        method = self.method_var.get()
        
        if method == 'yolo':
            self.ocr_frame.pack_forget()
            self.yolo_frame.pack(fill="x", padx=5, pady=5)
        elif method == 'ocr':
            self.yolo_frame.pack_forget()
            self.ocr_frame.pack(fill="x", padx=5, pady=5)
        else:  # hybrid
            self.yolo_frame.pack(fill="x", padx=5, pady=5)
            self.ocr_frame.pack(fill="x", padx=5, pady=5)
        
        if self.current_image_path:
            self.start_processing()
    
    def on_param_change_delayed(self, *args):
        """Обработчик изменения параметров с задержкой"""
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(self.update_delay, self.on_param_change)
    
    def on_param_change(self, *args):
        """Обработчик изменения параметров"""
        if self.current_image_path and self.detector:
            self.update_params()
            self.start_processing()
    
    def update_params(self):
        """Обновление параметров детектора"""
        if not self.detector:
            return
        
        # Собираем все параметры
        all_params = {
            'method': self.method_var.get(),
            'conf_threshold': self.conf_threshold_var.get(),
            'show_boxes': self.show_boxes_var.get(),
            'show_labels': self.show_labels_var.get(),
            'box_thickness': self.box_thickness_var.get(),
            'corner_padding': self.corner_padding_var.get(),
            'corner_size_factor': self.corner_size_var.get(),
            'rank_vertical_split': self.rank_split_var.get(),
            'show_corner': self.show_corner_var.get(),
            'show_roi': self.show_roi_var.get(),
            'show_text': self.show_text_var.get(),
            'min_aspect_ratio': self.min_aspect_var.get(),
            'max_aspect_ratio': self.max_aspect_var.get()
        }
        
        self.detector.set_params(all_params)
    
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            self.show_original()
            self.start_processing()
    
    def show_original(self):
        """Отображение оригинального изображения"""
        if self.current_image is not None:
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((canvas_width, canvas_height))
                
                self.original_photo = ImageTk.PhotoImage(img_pil)
                self.original_canvas.delete("all")
                
                x = (canvas_width - img_pil.width) // 2
                y = (canvas_height - img_pil.height) // 2
                self.original_canvas.create_image(x, y, image=self.original_photo, anchor=tk.NW)
            else:
                self.root.after(100, self.show_original)
    
    def start_processing(self):
        """Запуск обработки изображения"""
        if self.current_image_path and self.detector:
            # Проверяем, не выполняется ли уже обработка
            if self.detector.is_processing():
                print("⚠ Обработка уже выполняется")
                return
            
            # Показываем индикатор обработки
            self.progress_label.config(text="Обработка...", foreground="orange")
            
            # Обновляем параметры
            self.update_params()
            
            # Запускаем асинхронную обработку
            self.detector.detect_cards(
                self.current_image_path,
                async_mode=True,
                callback=self.on_processing_complete
            )
    
    def on_processing_complete(self, result_img, num_cards, stages, results):
        """Callback при завершении обработки"""
        self.result_image = result_img
        self.stages = stages
        
        # Отображение результата
        self.root.after(0, self.show_result)
        
        # Обновляем информацию
        info_text = f"Карт обнаружено: {num_cards}\n\n"
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Результаты распознавания:\n")
        
        for r in results:
            if r.get('name', 'Unknown') != "Unknown":
                info_text += f"Карта {r['index']}: {r['name']} "
                info_text += f"(уверенность: {r.get('confidence', 0)}%)\n"
                
                self.results_text.insert(tk.END, 
                    f"Карта {r['index']}: {r['name']} "
                    f"(уверенность: {r.get('confidence', 0)}%)\n")
        
        self.info_label.config(text=info_text)
        self.progress_label.config(text="Готово", foreground="green")
    
    def check_updates(self):
        """Периодическая проверка обновлений"""
        if self.detector and self.detector.check_for_updates():
            result_img, num_cards, stages = self.detector.get_last_result()
            results = self.detector.get_recognition_results()
            if result_img is not None:
                self.on_processing_complete(result_img, num_cards, stages, results)
        
        self.root.after(100, self.check_updates)
    
    def show_result(self):
        """Отображение результата обработки"""
        if self.result_image is not None:
            canvas_width = self.result_canvas.winfo_width()
            canvas_height = self.result_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img_rgb = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((canvas_width, canvas_height))
                
                self.result_photo = ImageTk.PhotoImage(img_pil)
                self.result_canvas.delete("all")
                
                x = (canvas_width - img_pil.width) // 2
                y = (canvas_height - img_pil.height) // 2
                self.result_canvas.create_image(x, y, image=self.result_photo, anchor=tk.NW)
            else:
                self.root.after(100, self.show_result)
    
    def save_params(self):
        """Сохранение параметров в файл"""
        if not self.detector:
            messagebox.showwarning("Предупреждение", "Детектор не инициализирован")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Сохранить параметры",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                for key, value in self.detector.params.items():
                    f.write(f"{key}={value}\n")
            messagebox.showinfo("Успех", "Параметры сохранены")
    
    def load_params(self):
        """Загрузка параметров из файла"""
        file_path = filedialog.askopenfilename(
            title="Загрузить параметры",
            filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            try:
                params = {}
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            # Преобразование типов
                            if value.replace('.', '').replace('-', '').isdigit():
                                if '.' in value:
                                    params[key] = float(value)
                                else:
                                    params[key] = int(value)
                            elif value.lower() == 'true':
                                params[key] = True
                            elif value.lower() == 'false':
                                params[key] = False
                            else:
                                params[key] = value
                
                # Обновляем переменные GUI
                if 'conf_threshold' in params:
                    self.conf_threshold_var.set(params['conf_threshold'])
                if 'show_boxes' in params:
                    self.show_boxes_var.set(params['show_boxes'])
                if 'show_labels' in params:
                    self.show_labels_var.set(params['show_labels'])
                if 'box_thickness' in params:
                    self.box_thickness_var.set(params['box_thickness'])
                if 'corner_padding' in params:
                    self.corner_padding_var.set(params['corner_padding'])
                if 'corner_size_factor' in params:
                    self.corner_size_var.set(params['corner_size_factor'])
                if 'rank_vertical_split' in params:
                    self.rank_split_var.set(params['rank_vertical_split'])
                if 'show_corner' in params:
                    self.show_corner_var.set(params['show_corner'])
                if 'show_roi' in params:
                    self.show_roi_var.set(params['show_roi'])
                if 'show_text' in params:
                    self.show_text_var.set(params['show_text'])
                if 'method' in params:
                    self.method_var.set(params['method'])
                    self.on_method_change()
                
                # Обновляем параметры детектора
                if self.detector:
                    self.detector.set_params(params)
                    if self.current_image_path:
                        self.start_processing()
                
                messagebox.showinfo("Успех", "Параметры загружены")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке параметров: {str(e)}")
    
    def save_result(self):
        """Сохранение результата в файл"""
        if self.result_image is not None:
            file_path = filedialog.asksaveasfilename(
                title="Сохранить результат",
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
            )
            if file_path:
                cv2.imwrite(file_path, self.result_image)
                messagebox.showinfo("Успех", "Результат сохранен")

def run_gui():
    """Функция для запуска GUI"""
    root = tk.Tk()
    app = CardDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()