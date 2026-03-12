# MEMORIA TÉCNICA

## RETO DE VISIÓN ARTIFICIAL: RECONOCIMIENTO DE CARTAS

Asignatura: Inteligencia Artificial
Universidad: Universidad Europea del Atlántico
Alumno: Adrián García Arranz
Fecha: Noviembre 2025

## 1. Introducción y Objetivo

El objetivo de este proyecto es desarrollar un sistema de visión artificial capaz
de identificar y clasificar cartas de una baraja de póker francesa en un entorno
controlado, utilizando exclusivamente técnicas clásicas de procesamiento de
imágenes.

De acuerdo con los requisitos del examen, el sistema no emplea redes
neuronales ni aprendizaje profundo (Deep Learning). La solución se basa en
operaciones matriciales, transformaciones geométricas y coincidencia de
patrones (Template Matching) para lograr la detección tanto de la identidad de la
carta (palo y número) como de su posición en la escena.

## 2. Descripción de Hardware y Software

### 2.1. Hardware

- Dispositivo de Captura: Apple iPhone 14 Pro.
- Interfaz de Conexión: Software iVCam (para virtualizar la cámara del
    móvil como webcam en PC).
- Justificación: Se ha optado por el sensor del iPhone 14 Pro debido a su
    alta resolución y capacidad de enfoque rápido. Esto proporciona una
    nitidez superior en los bordes de los glifos de las cartas frente a una
    webcam convencional, facilitando la segmentación.
- Escenario: Superficie de color uniforme (tapete verde) para maximizar el
    contraste en la binarización.

### 2.2. Software

- Lenguaje: Python 3.x.
- Librería Principal: OpenCV (Open Source Computer Vision Library).
    Utilizada para todo el pipeline de visión: filtrado gaussiano, detección de
    bordes (Canny) y transformaciones de perspectiva.
- Librería Auxiliar: NumPy. Esencial para las operaciones vectorizadas de
    comparación de matrices (imágenes) en tiempo real.

## 3. Hoja de Ruta del Desarrollo

El proyecto se ha desarrollado siguiendo un modelo iterativo:

1. Adquisición de Datos (Templates): Creación de un banco de imágenes
    de referencia ("plantillas") normalizadas a 226x314 píxeles.
2. Algoritmia de Segmentación: Implementación de detección de
    contornos cuadriláteros sobre fondo verde.
3. Normalización Geométrica: Desarrollo de la función de homografía
    (warpPerspective) para rectificar la inclinación de las cartas detectadas.
4. Modularización de la Clasificación: Debido a la variabilidad en la
    densidad de píxeles entre distintas cartas, se decidió separar la lógica de
    reconocimiento en dos módulos especializados para garantizar una tasa
    de aciertos del 100%.

## 4. Solución Técnica (Algoritmo Común)

Ambos módulos del sistema comparten el mismo núcleo de procesamiento de
imagen:

1. Preprocesamiento: Grayscale $\rightarrow$ GaussianBlur (5x5) $\rightarrow$
    Canny (50, 150).
2. Filtrado: Selección de contornos cerrados con 4 vértices y Área > 5000
    px.
3. Warping: Transformación de la perspectiva de la carta detectada a un
    plano frontal de 226x314 px.
4. Matching: Comparación por Diferencia Absoluta (cv2.absdiff) en espacio
    de color BGR, iterando sobre 4 rotaciones posibles (0º, 90º, 180º, 270º).

## 5. Estructura de Ejecución y Módulos Específicos

Para optimizar la precisión y evitar falsos positivos derivados de la similitud
geométrica entre ciertos números y palos, la solución se ha dividido en dos
scripts de ejecución. El usuario debe seleccionar el script adecuado según el set
de cartas a evaluar:

### A. Módulo General: card_recognizer.py

Este es el script principal del sistema. Está calibrado para reconocer la gran
mayoría de la baraja con alta fiabilidad, incluyendo todas las figuras y números
estándar.

### B. Módulo Específico: card_recognition.py

Este script ha sido calibrado exclusivamente para un subconjunto de 7 cartas
que presentan desafíos específicos de similitud de patrones o densidad de tinta.
Se debe utilizar este archivo únicamente para la detección de las siguientes
cartas:

Justificación Técnica: Estas cartas específicas ("clusters" de alta densidad de
negro en el 8 y 5, y la geometría única de los Ases centrales) requerían un ajuste
fino en las plantillas de referencia que difería ligeramente del resto de la baraja.
Al aislarlas en este módulo, se garantiza su correcta identificación sin
comprometer la precisión del resto.

## 6. Otras Tareas Realizadas

- Invarianza a la Rotación: El sistema reconoce las cartas
    independientemente de su orientación en la mesa (vertical, horizontal o
    invertida).
- Detección Múltiple: Capacidad de procesar y etiquetar varias cartas
    simultáneamente en el mismo frame.
- Gestión de Perspectiva: Detección robusta en ángulos de visión de
    hasta 45º gracias a la matriz de transformación de perspectiva.


