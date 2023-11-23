from page_detection import cleaned_image
from ocr_with_east import detect_localized_text
from word_processing import cleaned_words
import pytesseract

def wrapper(file, localize=False, min_len=3,closest_med=1):
    image = cleaned_image(file)
    
    #If needed, you can localize for scenic images where background is present
    if localize:
        text_from_image = detect_localized_text(image)[1]
        
    else:
        text_from_image = pytesseract.image_to_string(image)
        
    words_final = cleaned_words(text_from_image,min_len,closest_med)
    
    return words_final

def main():
    file= input("Enter Location of image")
    print(wrapper(file))

if __name__ == '__main__':
    main()
