import cv2
import sys
import os

# Добавляем путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ocr_recognizer import OCRCardRecognizer
from src.card_detector import CardDetector

def test_ocr_recognition():
    """Тестирование OCR распознавания"""
    
    # Путь к изображению
    image_path = "C:\\Users\\Dmitriy\\Desktop\\card_detector\\input\\aces.jpg"
    
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден!")
        print("Поместите изображение с картами в папку input/")
        return
    
    # Для Windows укажите путь к Tesseract
    tesseract_path = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    
    # Создаем детектор с распознаванием
    detector = CardDetector(tesseract_path=tesseract_path)
    
    # Включаем распознавание
    detector.params['use_recognition'] = True
    
    # Детектируем и распознаем карты
    print("Обработка изображения...")
    result_img, num_cards = detector.detect_cards(image_path)
    
    print(f"\nОбнаружено карт: {num_cards}")
    
    # Получаем детальные результаты
    if hasattr(detector, 'get_recognition_results'):
        results = detector.get_recognition_results()
        print("\nРезультаты распознавания:")
        for r in results:
            print(f"Карта {r['index']}: {r['name']} (уверенность: {r['confidence']}%)")
    
    # Показываем результат
    cv2.imshow("OCR Recognition Result", result_img)
    print("\nНажмите любую клавишу для закрытия...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Сохраняем результат
    output_path = "output/ocr_result.jpg"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, result_img)
    print(f"\nРезультат сохранен в: {output_path}")

if __name__ == "__main__":
    test_ocr_recognition()