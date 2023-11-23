import re
from autocorrect import Speller 
import nltk
from nltk.corpus import wordnet

def editDistance(s,t):
    n = len(s)
    m = len(t)

    prev = [j for j in range(m+1)]
    curr = [0] * (m+1)

    for i in range(1, n+1):
        curr[0] = i
        for j in range(1, m+1):
            if s[i-1] == t[j-1]:
                curr[j] = prev[j-1]
            else:
                mn = min(1 + prev[j], 1 + curr[j-1])
                curr[j] = min(mn, 1 + prev[j-1])
        prev = curr.copy()

    return prev[m]


def get_med_dict():
    word_dict = {}
    with open("med_terms.txt",'r') as file:
        lines = file.readlines()
        
    for line in lines:
        first_letter=line[0].lower()
        if first_letter in word_dict:
            word_dict[first_letter].append(line.strip())
        else:
            word_dict[first_letter] = [line.strip()]
            
    return word_dict


def get_closest(ocr_word, max_dist=3,length=5):
    first = ocr_word[0].lower()
    d = get_med_dict()
    
    med_list = d[first]
    closest = []
    for word in med_list:
        if editDistance(word,ocr_word) <=max_dist:
            closest.append(word)
            
    max_len = min(length,len(closest))
    return closest[:max_len]


def cleaned_words(text,min_len = 3,closest_med=5):
    
    ocr_words_0 = re.findall(r'[a-zA-Z]+', text)
    ocr_words = []
    
    for word in ocr_words_0:
        if len(word) >=min_len:
            ocr_words.append(word.lower())        
        
    spell = Speller()
    
    corrected = [spell(word) for word in ocr_words]
    english_words = []
    
    for word in corrected:
        synsets = wordnet.synsets(word)
        if synsets:
            english_words.append(word)
        else:
            english_words.append('')
            
    med_closest = [get_closest(ocr_word,closest_med) for ocr_word in ocr_words]
    
    key_words={}
    for i,word in enumerate(ocr_words):
        if word in key_words:
            key_words[word].append(english_words[i])
        else:
            key_words[word]=[english_words[i]]
            
        key_words[word].append(med_closest[i])
        
        
    return key_words


def main():
    pass

if __name__ == '__main__':
    main()