import os
import random
import shutil
import test
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from options.test_options import TestOptions


# def a function to overlay a alpha channel to a image
def img_transparent(img_path, alpha_path):
    #read the image
    img = Image.open(img_path)
    #read the alpha channel
    alpha = Image.open(alpha_path)
    #merge the image and alpha channel
    img.putalpha(alpha)
    return img

#def a function to overlay to images
def image_overlay(img1,img2):
    img_size = img1.shape

    img = Image.new('RGBA', (img_size[0], img_size[1]))
    img.paste(img1, (0, 0), img1)


# upper_triangular_matrix_copy to the lower triangular matrix    3d array
def upper_triangular_matrix_copy(arr):
    for i in range(len(arr)-1):
        arr[:, i, :] = arr[i, :, :]
    return arr

# lower_triangular_matrix_copy to the upper triangular matrix    3d array
def lower_triangular_matrix_copy(arr):
    for i in range(len(arr)-1):
        arr[i, :, :] = arr[:, i, :]
    return arr

# globe_crop_posion def：
#      1  2  3
#      4  5  6
#      7  8  9
def img_edge_transparent(img_path, padding_posion, globe_crop_posion = 0 ):
    #

    #read the image and resize it to 256*256 as np array
    img = cv2.imread(img_path)
    # sq_img = cv2.resize(img,(256,256))
    sq_img = cv2.resize(img,(img.shape[0],img.shape[0]))
    sq_img_copy = sq_img.copy()
    #change the color channel from BGR to RGB
    sq_img_copy[:,:,0] = sq_img[:,:,2]
    sq_img_copy[:,:,2] = sq_img[:,:,0]

    #raise error if the padding_posion is larger than the image size or small than 0 or not int
    if padding_posion > sq_img.shape[0] or padding_posion < 0 or type(padding_posion) != int:
        raise ValueError('padding_posion must be int and smaller than the image size')

    fig_size = sq_img.shape
    #create a 256*256*1 np array named transparent_tensor that all elements are 255, make right half of the transparent_tensor evenly reduce to 0, and all values are int
    transparent_tensor = np.ones((fig_size[0],fig_size[1],1),dtype=np.uint8)*255
    transparent_tensor_copy = transparent_tensor.copy()
    grad_trans_tensor_edge_right = np.ones((fig_size[0],padding_posion,1),dtype=np.uint8)*255
    grad_trans_tensor = np.ones((padding_posion,padding_posion,1),dtype=np.uint8)*255

    grad_trans_tensor_edge_right[:, -padding_posion:, :] = np.linspace(255,0,padding_posion).reshape(1,padding_posion,1)        #edge grad tensor
    grad_trans_tensor[:, -padding_posion:, :] = np.linspace(255,0,padding_posion).reshape(1,padding_posion,1)                   #conner grad tensor

    grad_trans_tensor_edge_bottom = np.rot90(grad_trans_tensor_edge_right,-1)                                                   #rotate the edge grad tensor
    grad_trans_tensor_edge_left = np.rot90(grad_trans_tensor_edge_bottom,-1)
    grad_trans_tensor_edge_top = np.rot90(grad_trans_tensor_edge_left,-1)

    grad_trans_tensor_conner_br = upper_triangular_matrix_copy(grad_trans_tensor)                                               #rotate the conner grad tensor
    grad_trans_tensor_conner_bl = np.rot90(grad_trans_tensor_conner_br,-1)
    grad_trans_tensor_conner_tl = np.rot90(grad_trans_tensor_conner_bl,-1)
    grad_trans_tensor_conner_tr = np.rot90(grad_trans_tensor_conner_tl,-1)
    if globe_crop_posion == 0:
        pass

    elif globe_crop_posion == 1:
        transparent_tensor[:, -padding_posion:, :] = grad_trans_tensor_edge_right                                               #right edge
        transparent_tensor[-padding_posion:, :, :] = grad_trans_tensor_edge_bottom                                              #bottom edge
        transparent_tensor[-padding_posion:, -padding_posion:, :] = grad_trans_tensor_conner_br                                 #conner right bottom

    elif globe_crop_posion == 2:
        transparent_tensor[:, -padding_posion:, :] = grad_trans_tensor_edge_right                                               #right edge
        transparent_tensor[-padding_posion:, :, :] = grad_trans_tensor_edge_bottom                                              #bottom edge
        transparent_tensor[:, :padding_posion, :] = grad_trans_tensor_edge_left                                                 #left edge
        transparent_tensor[-padding_posion:, -padding_posion:, :] = grad_trans_tensor_conner_br                                 #conner right bottom
        transparent_tensor[-padding_posion:, :padding_posion, :] = grad_trans_tensor_conner_bl                                  #conner left bottom        

    elif globe_crop_posion == 3:
        # transparent_tensor[-padding_posion:, :, :] = grad_trans_tensor_edge_bottom                                              #bottom edge
        transparent_tensor[:, :padding_posion, :] = grad_trans_tensor_edge_left                                                 #left edge
        # transparent_tensor[-padding_posion:, :padding_posion, :] = grad_trans_tensor_conner_bl                                  #conner left bottom

    elif globe_crop_posion == 4:
        transparent_tensor[:padding_posion, :, :] = grad_trans_tensor_edge_top                                                  #top edge
        transparent_tensor[:, -padding_posion:, :] = grad_trans_tensor_edge_right                                               #right edge
        transparent_tensor[-padding_posion:, :, :] = grad_trans_tensor_edge_bottom                                              #bottom edge
        transparent_tensor[:padding_posion, -padding_posion:, :] = grad_trans_tensor_conner_tr                                  #conner right top
        transparent_tensor[-padding_posion:, -padding_posion:, :] = grad_trans_tensor_conner_br                                 #conner right bottom

    elif globe_crop_posion == 5:
        transparent_tensor[:padding_posion, :, :] = grad_trans_tensor_edge_top                                                  #top edge
        transparent_tensor[:, -padding_posion:, :] = grad_trans_tensor_edge_right                                               #right edge
        transparent_tensor[-padding_posion:, :, :] = grad_trans_tensor_edge_bottom                                              #bottom edge
        transparent_tensor[:, :padding_posion, :] = grad_trans_tensor_edge_left                                                 #left edge    
        transparent_tensor[:padding_posion, -padding_posion:, :] = grad_trans_tensor_conner_tr                                  #conner right top
        transparent_tensor[-padding_posion:, -padding_posion:, :] = grad_trans_tensor_conner_br                                 #conner right bottom
        transparent_tensor[:padding_posion, :padding_posion, :] = grad_trans_tensor_conner_tl                                   #conner left top
        transparent_tensor[-padding_posion:, :padding_posion, :] = grad_trans_tensor_conner_bl                                  #conner left bottom

    elif globe_crop_posion == 6:
        transparent_tensor[:padding_posion, :, :] = grad_trans_tensor_edge_top                                                  #top edge
        transparent_tensor[:, :padding_posion, :] = grad_trans_tensor_edge_left                                                 #left edge    
        transparent_tensor[-padding_posion:, :, :] = grad_trans_tensor_edge_bottom                                              #bottom edge
        transparent_tensor[:padding_posion, :padding_posion, :] = grad_trans_tensor_conner_tl                                   #conner left top
        transparent_tensor[-padding_posion:, :padding_posion, :] = grad_trans_tensor_conner_bl                                  #conner left bottom

    elif globe_crop_posion == 7:
        transparent_tensor[:padding_posion, :, :] = grad_trans_tensor_edge_top                                                  #top edge
        # transparent_tensor[:, -padding_posion:, :] = grad_trans_tensor_edge_right                                               #right edge
        # transparent_tensor[:padding_posion, -padding_posion:, :] = grad_trans_tensor_conner_tr                                  #conner right top

    elif globe_crop_posion == 8:
        transparent_tensor[:padding_posion, :, :] = grad_trans_tensor_edge_top                                                  #top edge
        transparent_tensor[:, -padding_posion:, :] = grad_trans_tensor_edge_right                                               #right edge
        transparent_tensor[:, :padding_posion, :] = grad_trans_tensor_edge_left                                                 #left edge    
        transparent_tensor[:padding_posion, -padding_posion:, :] = grad_trans_tensor_conner_tr                                  #conner right top
        transparent_tensor[:padding_posion, :padding_posion, :] = grad_trans_tensor_conner_tl                                   #conner left top

    elif globe_crop_posion == 9:
        transparent_tensor[:padding_posion, :, :] = grad_trans_tensor_edge_top                                                  #top edge
        transparent_tensor[:, :padding_posion, :] = grad_trans_tensor_edge_left                                                 #left edge
        transparent_tensor[:padding_posion, :padding_posion, :] = grad_trans_tensor_conner_tl                                   #conner left top      

    # raise a error if the globe_crop_posion is not in the range of 0-9
    else:
        raise ValueError('globe_crop_posion must be in the range of 0-9')

    transparent_tensor = transparent_tensor.astype(np.uint8)



    # use the rgb_tensor as the background, and use the transparent_tensor as the alpha channel to create a 256*256*4 np array and save it as a png file by cv2
    # return np.concatenate([sq_img,transparent_tensor],axis=2)
    return Image.fromarray(np.concatenate([sq_img_copy,transparent_tensor],axis=2))

# five img past to a biger image
def five_crop_past(img_path_list):
    #check the image number, if it is not 5, raise a error
    if len(img_path_list) != 5:
        raise ValueError('the five_crop number of images must be 5')
    #get the image size
    img_size = Image.open(img_path_list[0]).size

    #set the padding position    
    padding_posion = int(img_size[0] * 1/3)

    t_img1 = img_edge_transparent(img_path_list[0], padding_posion, 0)
    
    img_size = t_img1.size

   
    t_img1.save('.\\transparency\\transparency_adjust\\t1.png')
    t_img2 = img_edge_transparent(img_path_list[1], padding_posion, 3)
    t_img2.save('.\\transparency\\transparency_adjust\\t2.png')
    t_img3 = img_edge_transparent(img_path_list[2], padding_posion, 7)
    t_img3.save('.\\transparency\\transparency_adjust\\t3.png')
    t_img4 = img_edge_transparent(img_path_list[3], padding_posion, 9)
    t_img4.save('.\\transparency\\transparency_adjust\\t4.png')
    t_img5 = img_edge_transparent(img_path_list[4], int(padding_posion/2), 5)
    t_img5.save('.\\transparency\\transparency_adjust\\t5.png')

    # paste the 4 transparent images together and save the alpha channel as a png file
    t_img = Image.new('RGBA', (img_size[0]*2-padding_posion, img_size[1]*2-padding_posion))
    t_img.paste(t_img1, (0, 0), t_img1)
    t_img.paste(t_img2, (img_size[0]-padding_posion, 0), t_img2)
    t_img.paste(t_img3, (0, img_size[1]-padding_posion), t_img3)
    t_img.paste(t_img4, (img_size[0]-padding_posion, img_size[1]-padding_posion), t_img4)
    t_img.paste(t_img5, (int((img_size[0]*2-padding_posion)/2)-int(img_size[0]/2),int((img_size[0]*2-padding_posion)/2)-int(img_size[0]/2)), t_img5)
    
    return t_img.convert('RGB')
    # t_img.convert('RGB').save('.\\transparency\\transparency_adjust\\big.png')




#crop_time only support 10 and 5
def ten_crop(img_path, crop_time = 5, padding_ratio=1/5):
    #check the crop_time, if it is not 10 or 5, raise a error
    if crop_time != 10 and crop_time != 5:
        raise ValueError('crop_time must be 10 or 5')
    
    padding_ratio = (padding_ratio + 1)/2
    #read the image
    img = cv2.imread(img_path)
    #get the image size
    img_size = img.shape
    #resize the image to square
    sq_img = cv2.resize(img, (img_size[0], img_size[0]))
    #crop the image to 5 parts, the size of every part padding_ratio of the original image, five parts are topleft, topright, bottomleft, bottomright, center
    sq_img_crop = [sq_img[:int(img_size[0]*padding_ratio), :int(img_size[0]*padding_ratio)], 
                   sq_img[:int(img_size[0]*padding_ratio), int(img_size[0]*(1-padding_ratio)):], 
                   sq_img[int(img_size[0]*(1-padding_ratio)):, :int(img_size[0]*padding_ratio)], 
                   sq_img[int(img_size[0]*(1-padding_ratio)):, int(img_size[0]*(1-padding_ratio)):], 
                   sq_img[int(img_size[0]*((1-padding_ratio)/2)):int(img_size[0]*((1+padding_ratio)/2)), int(img_size[0]*((1-padding_ratio)/2)):int(img_size[0]*((1+padding_ratio)/2))]]
    
    if crop_time == 10:
        #flip those 5 parts
        sq_img_crop_flip = [cv2.flip(sq_img_crop[0], 1), cv2.flip(sq_img_crop[1], 1), cv2.flip(sq_img_crop[2], 1), cv2.flip(sq_img_crop[3], 1), sq_img_crop[4]]
        return sq_img_crop + sq_img_crop_flip

    elif crop_time == 5:
        return sq_img_crop


def test_big_img(img_path):

        #big_img -> ten_crop -> ten_crop_plus
        ######################################################################################################
        img_list = ten_crop(img_path)
        for i in range(len(img_list)):
            cv2.imwrite('.\\transparency\\ten_crop\\{}.png'.format(i), img_list[i])
        
        crop_img_path_list = ['.\\transparency\\ten_crop\\{}.png'.format(i) for i in range(5)]
        for i in range(len(crop_img_path_list)):
            small_img_list = ten_crop(crop_img_path_list[i])
            for j in range(len(small_img_list)):
                cv2.imwrite('.\\transparency\\ten_crop_plus\\{}_{}.png'.format(i, j), small_img_list[j])
        ######################################################################################################


        #ten_crop_plus -> ten_crop_plus_G
        ######################################################################################################
        #copy all images of the ten_crop_plus to dataset/maps/testA
        for root, dirs, files in os.walk('.\\transparency\\ten_crop_plus'):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    shutil.copy(os.path.join(root, file), '.\\datasets\\maps\\testA')

        test.main()  #CycleGAN test

        #choose the images name with "fake" in the folder and move them to the folder ten_crop_plus_G
        result_path = '.\\results\\'+ opt.name +'\\test_latest\\images'

        for root, dirs, files in os.walk(result_path):
            for file in files:
                if "fake" in os.path.splitext(file)[0]:
                    shutil.copy(os.path.join(root, file), '.\\transparency\\ten_crop_plus_G')
        #clean the '.\\results\\mol_to_stm_pretrained(020ft_024_025_030ft_BS_Align_ft)\\test_latest\\images' folder
        for root, dirs, files in os.walk(result_path):
            for file in files:
                os.remove(os.path.join(root, file))

        ######################################################################################################




        #ten_crop_plus_G -> ten_crop_G
        ######################################################################################################
        #read all images path in the folder and save them in a list
        crop_img_path_list = []
        for root, dirs, files in os.walk('.\\transparency\\ten_crop_plus_G'):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    crop_img_path_list.append(os.path.join(root, file))
        
        #cut the crop_img_path_list every 5 elements and save them in a new list
        crop_img_path_list_list = [crop_img_path_list[i:i+5] for i in range(0, len(crop_img_path_list), 5)]
        for i in range(len(crop_img_path_list_list)):
            img = five_crop_past(crop_img_path_list_list[i])
            img.save('.\\transparency\\ten_crop_G\\{}.png'.format(i))

        ######################################################################################################
        
        
        
        #ten_crop_G  -> big_img_G
        ######################################################################################################
        #read all images path in ten_crop_G and save them in a list
        crop_img_path_list = []
        for root, dirs, files in os.walk('.\\transparency\\ten_crop_G'):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    crop_img_path_list.append(os.path.join(root, file))
        #use the five images in ten_crop_G to generate the big image by five_crop_past
        img = five_crop_past(crop_img_path_list)
        # img.save('.\\transparency\\big_img_G\\{}_GAN.png'.format(img_name))
        ######################################################################################################

        time_end = time.time()
        #print the time cost in seconds
        print('time cost', round(time_end - time_start), 's')

        return img






if __name__ == '__main__':
    time_start = time.time()
    opt = TestOptions().parse()
    
    ######################################################################################################
    #the whole process
    #big_img -> ten_crop -> ten_crop_plus -> ten_crop_plus_G -> ten_crop_G -> big_img_G
    ######################################################################################################

    #read all images path in the folder and save them in a list
    img_path_list = []
    for root, dirs, files in os.walk('.\\transparency\\big_img'):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                img_path_list.append(os.path.join(root, file))
    print(img_path_list)

    for i in tqdm(range(len(img_path_list))):
        # img_path = '.\\transparency\\big_img\\020.png'
        img_path = img_path_list[i]

        #get the img_path file name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img_G_STM = test_big_img(img_path)                                          # generate the big image
        img_G_STM.save('.\\transparency\\big_img_G\\{}_GAN.png'.format(img_name))

        #open the image and rotate it 90 degrees
        img_rotate90 = cv2.rotate(cv2.imread(img_path), cv2.ROTATE_90_COUNTERCLOCKWISE)
        #create a cache folder
        if not os.path.exists('.\\transparency\\cache'):
            os.makedirs('.\\transparency\\cache')
        #save the image in the cache folder
        cv2.imwrite('.\\transparency\\cache\\cache.png', img_rotate90)
        #read the image in the cache folder
        img90_path = '.\\transparency\\cache\\cache.png'

        img_G_STM_90 = test_big_img(img90_path)                                      # generate the big image in 90 degrees
        
        fig_size = img_G_STM_90.size
        transparent_tensor = np.ones((fig_size[0],fig_size[1],1),dtype=np.uint8)*127
        img_G_STM_90copy = img_G_STM_90.copy()
        # #change the color channel from BGR to RGB
        # img_G_STM_90copy[:,:,0] = img_G_STM_90[:,:,2]
        # img_G_STM_90copy[:,:,2] = img_G_STM_90[:,:,0]
        img_G_STM_90copy = Image.fromarray(np.concatenate([img_G_STM_90copy,transparent_tensor],axis=2))
        img_G_STM_90copy.save('.\\transparency\\big_img_G\\{}_GAN_90.png'.format(img_name))
        #rotate the img_G_STM_90copy 90 degrees anticlockwise
        img_G_STM_90copy = img_G_STM_90copy.rotate(270, expand=True)
        #creata a RGBA plot witch the size of the img_G_STM_90copy
        t_img = Image.new('RGBA', fig_size)
        img_G_STM = img_G_STM.convert('RGBA')
        img_G_STM_90copy = img_G_STM_90copy.convert('RGBA')
        t_img.paste(img_G_STM, (0, 0), img_G_STM)
        t_img.paste(img_G_STM_90copy, (0, 0), img_G_STM_90copy)
        t_img = t_img.convert('RGB')
        t_img.save('.\\transparency\\big_img_G\\{}_GAN_tencrop.png'.format(img_name))







    


