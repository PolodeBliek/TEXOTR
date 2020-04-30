from PIL import Image
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
File = "C:\\Users\\Polo\\Documents\\GitHub\\OTR---LaTeX\\Testing\\"
print(pytesseract.image_to_string(Image.open(File + 'Pic0_0.png'), lang = "nld"))
