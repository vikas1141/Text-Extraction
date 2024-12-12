import fitz  
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import pytesseract


file_path = 'Doc3.pdf'
images_path = 'processed_images/'
os.makedirs(images_path, exist_ok=True)

pdf_file = fitz.open(file_path)
page_nums = len(pdf_file)


images_list = []
for page_num in range(page_nums):
    page_content = pdf_file[page_num]
    images_list.extend(page_content.get_images())


if len(images_list) == 0:
    print(f'No images found in {file_path}')
else:
    for i, img in enumerate(images_list, start=1):
        xref = img[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image['image']
        image_ext = base_image['ext']

        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

    
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = image_cv[y:y+h, x:x+w]
        else:
            cropped_image = image_cv 

        resized_image = cv2.resize(cropped_image, (800, 400))

      
        image_name = f"{i}_processed.{image_ext}"
        output_path = os.path.join(images_path, image_name)
        cv2.imwrite(output_path, resized_image)
        print(f"Processed and saved image {image_name}.")

      
        text = pytesseract.image_to_string(resized_image)
        print(f"Extracted Text from Image {i}:")
        print(text)

pdf_file.close()