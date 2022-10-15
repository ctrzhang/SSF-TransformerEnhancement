from curses.ascii import isdigit
import numpy as np
from glob import glob
import os
import cv2
from tqdm import tqdm

mesh_dict = {'F':1,'M':0,'Y':1,'N':0,'Working':0,'Security staff':0.1,'Commercial associate':0.2,'High skill tech staff':0.3,'Pensioner':0.4,'State servant':0.6,
'Student':0.8,'Accountants':1,'Cleaning staff':0.5,'Secondary / secondary special':0,'Higher education':0.2,'Incomplete higher':0.4,'Lower secondary':0.6,
'Academic degree':0.8,'Married':0,'Single / not married':0.2,'Civil marriage':0.4,'Separated':0.6,'Widow':0.8,
'House / apartment':0.2,'With parents':0.4,'Municipal apartment':0.6,'Rented apartment':0.8,'Office apartment':1,'Co-op apartment':0,
'Laborers':0.2,'Core staff':0.4,'Sales staff':0.6,'Managers':0.8,'Drivers':1}


works = ['', 'Security staff', 'Sales staff', 'Accountants', 'Laborers', 'Managers', 'Drivers', 'Core staff', 'High skill tech staff', 'Cleaning staff', 'Private service staff', 'Cooking staff', 'Low-skill Laborers', 'Medicine staff', 'Secretaries', 'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff']
def is_number(s):
    try:  
        float(s)
        return True
    except ValueError:  
        pass 
    try:
        import unicodedata  
        unicodedata.numeric(s)  
        return True
    except (TypeError, ValueError):
        pass
    return False


def reformat(csvfile):
    cleaned_record = []
    progress = 0
    with open(csvfile,'r') as f:
        line = f.readline()
        line = f.readline()
        while line:
            progress += 1
            temporal_list = []
            items = line.split(',')
            for index, item in enumerate(items):
                if index == 0:
                    continue

                if len(item)==0:
                    temporal_list.append(0)
                elif index==16:
                    temporal_list.append(works.index(item))
                elif not is_number(item):
                    temporal_list.append(mesh_dict[item])
                else:
                    temporal_list.append(abs(float(item)))

            line = f.readline()
            print(f'\r Step1: Cleaning - Executing line {progress}',end='', flush=True)
            cleaned_record.append(temporal_list)

    min_max_array = []
    for index in range(0,17):
        min_bound = 100000000
        max_bound = -100000000
        progress = 0
        for line in cleaned_record:
            progress += 1
            for idx, item in enumerate(line):
                if idx == index:
                    if item>max_bound:
                        max_bound = item
                    if item<min_bound:
                        min_bound=item

            print(f'\r Step2: Find min max - Round {index}:16 Executing line {progress}',end='', flush=True)
        min_max_array.append((min_bound,max_bound))

    print(f'\n generated min max array for each column:\n{min_max_array}')

    return cleaned_record,min_max_array


def generate_save_grayscale_image(sequence,min_max_array):
    imgs = []
    
    for index, item in enumerate(sequence):
        
        color_value = 0

        if item==1 or item==0:
            color_value = item*255
        elif abs(min_max_array[index][1]-min_max_array[index][0])==0:
            color_value = 0
        else:
            color_value = 255 * int(item-min_max_array[index][0])/abs(min_max_array[index][1]-min_max_array[index][0])

        generated_image = np.ones([20, 20], dtype=np.uint8)*color_value
        imgs.append(generated_image)
        pass
    
    im0 = imgs[0]
    for index, im in enumerate(imgs):
        if index==0:continue

        im0 = np.concatenate((im0, im), axis = 1)

    im0 = im0.astype(np.uint8)
    return im0

if __name__ == '__main__':
    csvfile = 'application_record.csv'

    if not os.path.exists(f'./generated_dataset/'):
        os.mkdir(f'./generated_dataset/') 

    x_array,min_max_array = reformat(csvfile)

    print(f'\n Step3: Executing color match and image generation: Total samples {len(x_array)}')

    imageid=0
    for line in tqdm(x_array):
        imageid+=1
        if imageid%50!=0:
            continue # MUST DELETE THIS!!!
        output_image = generate_save_grayscale_image(line,min_max_array)

        cv2.imwrite(f'./generated_dataset/{imageid}.png',output_image)
        
