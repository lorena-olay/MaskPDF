import io
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import pytesseract
import cv2
import numpy as np
import os

# Configurar la ruta del ejecutable de Tesseract si no está en el PATH
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

def enhance_image(image):
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(2)  # Mejora del contraste
    return enhanced_image

def mask_sensitive_data(image):
    # Convertir la imagen a formato compatible con OpenCV
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convertir RGB a BGR

    # Convertir a escala de grises
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # Cargar el modelo preentrenado de detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Enmascarar rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(open_cv_image, (x, y), (x+w, y+h), (0, 0, 0), -1)

    # Convertir la imagen de nuevo a formato PIL
    masked_image = Image.fromarray(open_cv_image[:, :, ::-1])
    return masked_image

def process_pdf(pdf_path, output_path):
    images = extract_images_from_pdf(pdf_path)
    for index, image in enumerate(images):
        enhanced_image = enhance_image(image)
        masked_image = mask_sensitive_data(enhanced_image)
        masked_image.save(f"{output_path}/page_{index}.png")

# Ejemplo de uso
pdf_path = "dni.pdf"
output_path = os.getcwd()
process_pdf(pdf_path, output_path)
