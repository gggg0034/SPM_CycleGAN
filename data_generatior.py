import math
import os
import random
import shutil
import time

from PIL import Image
from tqdm import tqdm

# path = r'C:/Users/73594/Desktop/my python/style transfer/pytorch-CycleGAN-and-pix2pix-master - test/datasets/maps/testA'
# newpath = r'C:/Users/73594/Desktop/my python/style transfer/pytorch-CycleGAN-and-pix2pix-master - test/datasets/maps/testA'

'''def a function to create cache folders, the folder {current path}/cache/rotated/trainA and {current path}/cache/rotated/trainB,
    {current path}/cache/flip/trainA and {current path}/cache/flip/trainB,
    {current path}/cache/crop/trainA and {current path}/cache/crop/trainB'''
def create_cache_folder(path):
    cache_path = os.path.join(path, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    rotated_path = os.path.join(cache_path, 'rotated')
    if not os.path.exists(os.path.join(rotated_path, 'trainA')):
        os.mkdir(rotated_path)
        os.mkdir(os.path.join(rotated_path, 'trainA'))
        os.mkdir(os.path.join(rotated_path, 'trainB'))
    flip_path = os.path.join(cache_path, 'flip')
    if not os.path.exists(os.path.join(flip_path, 'trainA')):
        os.mkdir(flip_path)
        os.mkdir(os.path.join(flip_path, 'trainA'))
        os.mkdir(os.path.join(flip_path, 'trainB'))
    crop_path = os.path.join(cache_path, 'crop')
    if not os.path.exists(os.path.join(crop_path, 'trainA')):
        os.mkdir(crop_path)
        os.mkdir(os.path.join(crop_path, 'trainA'))
        os.mkdir(os.path.join(crop_path, 'trainB'))
    
    return cache_path, rotated_path, flip_path, crop_path

'''def a function to align the images pix sizes in the two folders, such as the pix size of first image in folder A should align to the pix size of first image in folder B.'''
def align_pix_size(pathA, pathB):
    print('align pictures')
    filesA = os.listdir(pathA)
    filesB = os.listdir(pathB)
    for i in range(len(filesA)):
        files_A = os.path.join(pathA, filesA[i])
        files_B = os.path.join(pathB, filesB[i])
        imgA = Image.open(files_A)
        imgB = Image.open(files_B)
        if imgA.size != imgB.size:
            imgA = imgA.resize(imgB.size)
            imgA.save(files_A)




def tran32bit_to_24bit(path, newpath):
    files = os.listdir(path)
    for i in files:
        files = os.path.join(path, i)
        img = Image.open(files).convert('RGB')
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.png')
        img.save(dst)


#def a function to flip the picture upside down
def flip_picture_tb(path, newpath):
    files = os.listdir(path)
    for i in files:
        files = os.path.join(path, i)
        img = Image.open(files)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '_tb.png')
        img.save(dst)

#def a function to flip the picture left to right and rename the picture
def flip_picture_lr(path, newpath):
    files = os.listdir(path)
    print('flipping pictures')
    for i in tqdm((files)):
        files = os.path.join(path, i)
        img = Image.open(files)
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.png')
        img.save(dst) 
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '_lr.png')
        img.save(dst)

#def a function to rotate the picture 90, 180, 270 degree than rename those pictures
def rotate_picture(path, newpath):
    files = os.listdir(path)
    print('rotating pictures')
    for i in tqdm(files):
        files = os.path.join(path, i)
        img = Image.open(files)
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.png')
        img.save(dst)        
        img = img.rotate(90, expand=True)
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '_90.png')
        img.save(dst)
        img = img.rotate(90, expand=True)
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '_180.png')
        img.save(dst)
        img = img.rotate(90, expand=True)
        dirpath = newpath
        file_name, file_extend = os.path.splitext(i)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '_270.png')
        img.save(dst)


#def a function to rotate the image and shrink the image
def rotate_image(image, angle):
    #get the width and height of the image
    width, height = image.size
    #get the radian of the angle
    rad = math.radians(angle)
    #get the coefficient of expansion
    coefficient_of_expansion = math.fabs(math.sin(rad))+ math.fabs(math.cos(rad))
    #get the new width and height of the image
    new_width = int(width * (1/coefficient_of_expansion))
    new_height = int(height * (1/coefficient_of_expansion))
    #get the new image
    new_image = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0))
    #get the center of the image
    lift_top = (int(0 - width * (coefficient_of_expansion-1/coefficient_of_expansion) // 2), int(0 - height * (coefficient_of_expansion-1/coefficient_of_expansion) // 2))
    #rotate the image
    new_image.paste(image.rotate(angle, expand=1), lift_top)
    #return the new image
    return new_image


#def a function to ramdomly crop a square area to picture and rename the picture
def pair_crop_picture(imageA, imageB, angle_align = True):
    width, height = imageA.size
    #resize the imageB to the same size as imageA
    imageB = imageB.resize((width, height), Image.ANTIALIAS)
    min_value = min(width, height)
    if width*2<height or height*2<width:
        random_square = random.randint(int(min_value / 1), min_value)
    else:
        random_square = random.randint(int(min_value / 3), min_value)
    left = random.randint(0, width - random_square)
    top = random.randint(0, height - random_square)
    crop_imgA = imageA.crop((left, top, left + random_square, top + random_square))
    crop_imgB = imageB.crop((left, top, left + random_square, top + random_square))
    if angle_align:#rotate together
        angle = random.randint(0, 45)
        crop_imgA = rotate_image(crop_imgA, angle)
        crop_imgB = rotate_image(crop_imgB, angle)
    else:          #rotate independently
        crop_imgA = rotate_image(crop_imgA, random.randint(0, 45))
        crop_imgB = rotate_image(crop_imgB, random.randint(0, 45))

    return crop_imgA, crop_imgB

def data_generatior(imgA_path, imgB_path, new_imgA_path ,new_imgB_path, crop_times):

    imgA_name = os.listdir(imgA_path)
    imgB_name = os.listdir(imgB_path)
    #check if the number of imageA and imageB is the same
    if len(imgA_name) == len(imgB_name):
        print('randomly cropping pictures')
        for i in tqdm(range(len(imgA_name))):
            imgA = os.path.join(imgA_path, imgA_name[i])
            imgB = os.path.join(imgB_path, imgB_name[i])
            imgA = Image.open(imgA)
            imgB = Image.open(imgB)
            #resize the imageB to the same size as imageA
            imgB = imgB.resize(imgA.size, Image.ANTIALIAS)
            for j in range(crop_times):
                crop_imgA, crop_imgB = pair_crop_picture(imgA, imgB)
                dirpath = new_imgA_path
                file_name, file_extend = os.path.splitext(imgA_name[i])
                dst = os.path.join(os.path.abspath(dirpath), file_name + '_crop_' + str(j) + '.png')
                crop_imgA.save(dst)
                dirpath = new_imgB_path
                file_name, file_extend = os.path.splitext(imgB_name[i])
                dst = os.path.join(os.path.abspath(dirpath), file_name + '_crop_' + str(j) + '.png')
                crop_imgB.save(dst)
    else:
        #raise an error if the number of imageA and imageB is not the same
        raise ValueError('the number of imageA and imageB is not the same')
    

'''def a function to ramdomly shaffule the pictures in the folderA and shaffule the same squnce in the folderB'''
def shuffle_picture(pathA, pathB, newpathA, newpathB):
    filesA_list = os.listdir(pathA)
    filesB_list = os.listdir(pathB)
    #check if the number of imageA and imageB is the same
    if len(filesA_list) == len(filesB_list):
        c = list(zip(filesA_list, filesB_list))
        random.shuffle(c)
        filesA_list, filesB_list = zip(*c)
        print('shuffling pictures')
        for i in tqdm((range(len(filesA_list)))):
            filesA = os.path.join(pathA, filesA_list[i])
            filesB = os.path.join(pathB, filesB_list[i])
            imgA = Image.open(filesA)
            imgB = Image.open(filesB)
            dst = os.path.join(os.path.abspath(newpathA), str(i+1) + '_A.png')
            imgA.save(dst)
            dst = os.path.join(os.path.abspath(newpathB), str(i+1) + '_B.png')
            imgB.save(dst)
    else:
        #raise an error if the number of imageA and imageB is not the same
        raise ValueError('the number of imageA and imageB is not the same')

'''def a function to resize the pictures in the folderA 600pix*600pix and resize the same squnce in the folderB,
than stitch two pictures together as 1200pix*600pix and save it in the newpath'''
def stitch_picture(pathA, pathB, newpath):
    filesA_list = os.listdir(pathA)
    filesB_list = os.listdir(pathB)
    #check if the number of imageA and imageB is the same
    if len(filesA_list) == len(filesB_list):
        print('resizing pictures')
        for i in tqdm((range(len(filesA_list)))):
            filesA = os.path.join(pathA, filesA_list[i])
            filesB = os.path.join(pathB, filesB_list[i])
            imgA = Image.open(filesA)
            imgB = Image.open(filesB)
            imgA = imgA.resize((600, 600), Image.ANTIALIAS)
            imgB = imgB.resize((600, 600), Image.ANTIALIAS)
            new_img = Image.new('RGB', (1200, 600))
            new_img.paste(imgA, (0, 0))
            new_img.paste(imgB, (600, 0))
            dst = os.path.join(os.path.abspath(newpath), str(i+1) + '.png')
            new_img.save(dst)
    else:
        #raise an error if the number of imageA and imageB is not the same
        raise ValueError('the number of imageA and imageB is not the same')




if __name__ == '__main__':
    #get time
    start_time = time.time()
    #get current path
    current_path = os.path.join(os.getcwd(),'data_generation')   
    #create a cache_folder
    cache_path, rotated_path, flip_path, crop_path = create_cache_folder(os.path.join(current_path, 'image'))
    
    imgA_path = os.path.join(current_path, 'image', 'CPK_mol')
    imgB_path = os.path.join(current_path,  'image', 'STM_mol')


    new_imgA_path = os.path.join(current_path,  'image', 'random', 'trainA')
    new_imgB_path = os.path.join(current_path,  'image', 'random', 'trainB')
    new_img_path = os.path.join(current_path,  'image', 'random', 'train')  #Align dataset folder
    

    
    rotated_path_A = os.path.join(rotated_path, 'trainA')
    rotated_path_B = os.path.join(rotated_path, 'trainB')
    flip_path_A = os.path.join(flip_path, 'trainA')
    flip_path_B = os.path.join(flip_path, 'trainB')
    crop_path_A = os.path.join(crop_path, 'trainA')
    crop_path_B = os.path.join(crop_path, 'trainB')


    # #check whether the image in the folder is 24bit
    tran32bit_to_24bit(imgA_path, imgA_path)
    # check whether the folder is exist
    if os.path.exists(new_imgA_path) == False:
        os.mkdir(new_imgA_path)
    if os.path.exists(new_imgB_path) == False:
        os.mkdir(new_imgB_path)
    if os.path.exists(new_img_path) == False:
        os.mkdir(new_img_path)                 # gerenarte Align dataset folder
    print('start data generation')
    
    #align the pix_size of two folder
    align_pix_size(imgA_path, imgB_path)
    #frist rotate 
    rotate_picture(imgA_path, rotated_path_A)
    rotate_picture(imgB_path, rotated_path_B)
    #flip the picture
    flip_picture_lr(rotated_path_A, flip_path_A)
    flip_picture_lr(rotated_path_B, flip_path_B) 
    #crop the picture
    crop_times = 20
    data_generatior(flip_path_A, flip_path_B, crop_path_A ,crop_path_B, crop_times)
    #shuffle the picture
    shuffle_picture(crop_path_A,crop_path_B,new_imgA_path,new_imgB_path)
    #gerenarte Align dataset
    stitch_picture(new_imgA_path, new_imgB_path, new_img_path)
    
    
    #copy the folder to the same path and rename it as testA and testB
    shutil.copytree(new_imgA_path, os.path.join(current_path,  'image', 'random', 'testA'))
    shutil.copytree(new_imgB_path, os.path.join(current_path,  'image', 'random', 'testB'))
    shutil.copytree(new_img_path, os.path.join(current_path,  'image', 'random', 'test'))
    #copy the folder to the same path and rename it as valA and valB
    shutil.copytree(new_imgA_path, os.path.join(current_path,  'image', 'random', 'valA'))
    shutil.copytree(new_imgB_path, os.path.join(current_path,  'image', 'random', 'valB'))
    shutil.copytree(new_img_path, os.path.join(current_path,  'image', 'random', 'val'))
    #delete the cache folder
    shutil.rmtree(cache_path)
    #get time
    end_time = time.time()

    #print the time as int seconds
    print('time cost: ', str(round(end_time - start_time)), 's')
    # 4. Make this code easier to read, including by adding comments, renaming variables, and/or reorganizing the code.
    print('data generation finished')