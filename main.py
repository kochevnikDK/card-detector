#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import argparse
from src.card_detector import CardDetector
from src.gui_calibrator import run_gui

def main():
    parser = argparse.ArgumentParser(description='Детектирование игральных карт на изображении')
    parser.add_argument('--input', '-i', type=str,
                       help='Путь к входному изображению')
    parser.add_argument('--output', '-o', type=str, default='output/result.jpg',
                       help='Путь для сохранения результата')
    parser.add_argument('--method', '-m', type=str, choices=['threshold', 'canny'],
                       default='threshold', help='Метод детектирования')
    parser.add_argument('--params', '-p', type=str,
                       help='Путь к файлу с параметрами')
    parser.add_argument('--gui', '-g', action='store_true',
                       help='Запустить графический интерфейс для калибровки')
    
    args = parser.parse_args()
    
    if args.gui:
        # Запуск GUI
        run_gui()
        return
    
    if not args.input:
        print("Ошибка: Не указан путь к входному изображению. Используйте --input или --gui для запуска интерфейса.")
        return
    
    # Проверяем существование входного файла
    if not os.path.exists(args.input):
        print(f"Ошибка: Файл {args.input} не найден!")
        return
    
    # Создаем директорию для выходных файлов, если её нет
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Создаем экземпляр детектора
    detector = CardDetector()
    
    # Загружаем параметры из файла, если указан
    if args.params and os.path.exists(args.params):
        try:
            params = {}
            with open(args.params, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        if value.replace('.', '').isdigit():
                            if '.' in value:
                                params[key] = float(value)
                            else:
                                params[key] = int(value)
                        else:
                            params[key] = value
            detector.set_params(params)
            print(f"Параметры загружены из {args.params}")
        except Exception as e:
            print(f"Ошибка при загрузке параметров: {e}")
    
    # Устанавливаем метод детектирования
    detector.set_params({'method': args.method})
    
    try:
        # Обрабатываем изображение
        result_img, num_cards = detector.detect_cards(args.input)
        
        # Сохраняем результат
        cv2.imwrite(args.output, result_img)
        
        print(f"Обнаружено карт: {num_cards}")
        print(f"Результат сохранен в: {args.output}")
        
        # Показываем изображение
        cv2.imshow('Detected Cards', result_img)
        print("Нажмите любую клавишу для закрытия окна...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")

if __name__ == "__main__":
    main()