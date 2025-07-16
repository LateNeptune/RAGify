import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from pdf2image import convert_from_path
import tempfile
import os
from typing import List, Tuple
import streamlit as st

# Add Poppler to PATH if not already there
poppler_path = r"C:\Users\temp_dude\Downloads\poppler-24.08.0\Library\bin"
if poppler_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = poppler_path + os.pathsep + os.environ.get('PATH', '')

class HandwritingOCR:
    def __init__(self):
        """Initialize the handwriting OCR pipeline with CRAFT + TrOCR"""
        print("Loading models... This might take a moment on first run!")

        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Initialize EasyOCR (includes CRAFT for text detection)
        # Force GPU usage for EasyOCR if available
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("🚀 Initializing EasyOCR with GPU acceleration...")
        else:
            print("⚠️ GPU not available, using CPU for EasyOCR")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

        # Initialize TrOCR for handwriting recognition
        print("📝 Loading TrOCR models...")
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        # Move TrOCR model to GPU if available
        self.model = self.model.to(self.device)

        # Enable GPU optimizations if available
        if torch.cuda.is_available():
            self.model = self.model.half()  # Use half precision for faster inference
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

        print("Models loaded successfully!")

    def detect_text_regions(self, image):
        """Use CRAFT (via EasyOCR) to detect text regions"""
        # EasyOCR returns results as [bbox, text, confidence]
        results = self.reader.readtext(image)

        text_regions = []
        for (bbox, text, confidence) in results:
            # Convert bbox to format we can use
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            text_regions.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'confidence': confidence,
                'detected_text': text
            })

        return text_regions

    def recognize_handwriting(self, image_crop):
        """Use TrOCR to recognize handwritten text"""
        try:
            # Convert to PIL Image if needed
            if isinstance(image_crop, np.ndarray):
                image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

            # Process with TrOCR and move tensors to GPU
            pixel_values = self.processor(image_crop, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)  # Move to GPU

            # Convert to half precision if using GPU
            if torch.cuda.is_available():
                pixel_values = pixel_values.half()

            # Generate text using GPU
            with torch.no_grad():  # Disable gradient computation for inference
                generated_ids = self.model.generate(pixel_values)

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return generated_text
        except Exception as e:
            print(f"Error in TrOCR recognition: {e}")
            return ""

    def process_image(self, image_path: str) -> str:
        """Complete pipeline: detect + recognize text in an image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return ""

        # Detect text regions
        text_regions = self.detect_text_regions(image)

        all_text = []
        for region in text_regions:
            x_min, y_min, x_max, y_max = region['bbox']

            # Crop the text region
            cropped = image[y_min:y_max, x_min:x_max]

            # Use TrOCR for handwritten text recognition
            trocr_text = self.recognize_handwriting(cropped)

            # Use the better result (TrOCR for handwriting, EasyOCR as fallback)
            if trocr_text.strip():
                all_text.append(trocr_text)
            else:
                all_text.append(region['detected_text'])

        return ' '.join(all_text)

    def process_pdf(self, pdf_path: str) -> str:
        """Convert PDF to images and process each page"""
        try:
            # Try multiple methods to convert PDF to images
            pages = None
            poppler_paths = [
                r"C:\Users\temp_dude\Downloads\poppler-24.08.0\Library\bin",
                r"C:\poppler\Library\bin",
                r"C:\Program Files\poppler\bin",
                None  # Try without explicit path
            ]

            for poppler_path in poppler_paths:
                try:
                    if poppler_path:
                        print(f"Trying poppler path: {poppler_path}")
                        pages = convert_from_path(pdf_path, poppler_path=poppler_path)
                    else:
                        print("Trying without explicit poppler path")
                        pages = convert_from_path(pdf_path)
                    break  # Success, exit the loop
                except Exception as e:
                    print(f"Failed with poppler path {poppler_path}: {e}")
                    continue

            if pages is None:
                return "Error: Could not convert PDF to images. Please ensure Poppler is installed and in PATH."

            all_text = []
            for i, page in enumerate(pages):
                print(f"Processing page {i+1}/{len(pages)}")

                # Save page as temporary image with better file handling
                temp_file = None
                try:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_file_path = temp_file.name
                    temp_file.close()  # Close the file handle before saving

                    # Save the page image
                    page.save(temp_file_path, 'PNG')

                    # Process the image
                    print(f"  🔍 Analyzing page {i+1} for text...")
                    page_text = self.process_image(temp_file_path)
                    print(f"  📝 Extracted {len(page_text)} characters from page {i+1}")
                    if page_text.strip():  # Only add non-empty text
                        all_text.append(page_text)
                        print(f"  ✅ Added text from page {i+1}")
                    else:
                        print(f"  ⚠️ No text found on page {i+1}")

                except Exception as e:
                    print(f"Error processing page {i+1}: {e}")
                finally:
                    # Clean up temp file
                    if temp_file and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            print(f"Warning: Could not delete temp file {temp_file_path}: {e}")

            result = '\n\n'.join(all_text)
            print(f"🎉 Processing complete! Total text length: {len(result)} characters")
            if not result.strip():
                print("Warning: No text was extracted from the PDF")
                return "No readable text found in this document."
            print("✅ Returning extracted text to main app")
            return result
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return f"Error processing PDF: {str(e)}"