__author__ = 'sz372'

import cv2
import os

_decoder = cv2.text.OCRTesseract_create()

def ocr(roi):
    # Temporarily disable stderr
    # Tesseract prints out annoying error messages
    tmpfd = os.dup(2)
    os.close(2)
    text, comp_rect, comp_texts, comp_conf = _decoder.run(roi)
    os.dup2(tmpfd, 2)
    os.close(tmpfd)
    #text = text[0:-2] # remove newline added by tesseract
    return text
