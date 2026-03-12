#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.card_detector import CardDetector
from src.template_matcher import TemplateMatcher

def create_templates_from_image():
    """Создание шаблонов из изображения с картами"""
    
    image_path = input("C:\\Users\\Dmitriy\\Desktop\\card_detector\\input\\cards.png").strip()
    
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден!")
        return
    
    # Детектируем карты
    detector = CardDetector()
    result_img, num_cards = detector.detect_cards(image_path)
    
    if num_cards == 0:
        print("Карты не обнаружены на изображении!")
        return
    
    print(f"Обнаружено {num_cards} карт")
    
    # Создаем шаблоны
    matcher = TemplateMatcher()
    img = cv2.imread(image_path)
    matcher.create_templates_from_image(img, detector.last_contours)
    
    print("Шаблоны успешно созданы!")

def main():
    print("=" * 50)
    print("Инструмент для создания шаблонов карт")
    print("=" * 50)
    print("\nЭтот инструмент создаст шаблоны рангов и мастей")
    print("из изображения с картами для улучшения распознавания.\n")
    
    create_templates_from_image()

if __name__ == "__main__":
    main()