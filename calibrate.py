#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Запуск калибратора детектора игральных карт с графическим интерфейсом
"""

import sys
import os
# Добавляем текущую папку в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui_calibrator import run_gui

if __name__ == "__main__":
    print("Запуск калибратора детектора игральных карт...")
    run_gui()