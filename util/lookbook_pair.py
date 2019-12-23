# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 11:05:38 2019

@author: arden.zhu

Generate pairs images for pix2pix from lookbook images
"""

import os
import re
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.util import pad, img_as_ubyte

class Options():
    n_train = 400                            # number of product in training set
    n_test = 100                             # number of product in test set
    n_pair = 1                               # how many pair (maximum) for each product
    
    src_folder = 'c:/ml/lookbook/'           # folder of lookbook images
    target_folder = 'c:/ml/lookbook_paired/' # parent folder of lookbook pairs images
    job_name = 'test1'                       # name of job, result of job will put into {target_folder}/{job_name}
    
    image_size = 256                         # default paired image size is 512 * 256
    
    @property
    def job_target_folder(self):
        return self.target_folder + self.job_name + '/'
    
def create_target_path(opt):
    image_path = opt.job_target_folder + 'image/'
    if not os.path.exists(image_path):
        os.makedirs(image_path)    
        
    sub_path = image_path + 'train/'
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)

    sub_path = image_path + 'test/'
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
        

def get_lookbook_files(opt):
    '''
    return file names in lookbook
    '''
    image_path = opt.src_folder + 'data/'
    
    files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    return files

def read_lookbook(files):
    '''
    read files of lookbook images and return the look book structure
    Image file name: sprintf( 'PID%06d_CLEAN%d_IID%06d', product_id, is_product_image, image_id )
    
    files : a list of string
    returns:
        a dictionary of 
        {'product_id': '000012', 
         'product_image': 'PID000000_CLEAN1_IID000011',
         'model_images': ['PID000000_CLEAN0_IID000000', ......]}
    
    '''
    # a regex to get info from file name
    p = re.compile(r'PID(?P<product_id>\d{6})_CLEAN(?P<is_product>\d)_IID(?P<image_id>\d{6})')
    
    # a dictionary of products
    dic = {}
    
    for f in files:
        m = p.match(f)
        if m:
            fi = m.groupdict()  # fi is like {'product_id': '000012', 'is_product': '0', 'image_id': '000020'}
            if not (fi['product_id'] in dic):
                dic[fi['product_id']] = {'product_id': fi['product_id'], 
                                         'product_image': None, 
                                         'model_images': []}
            product = dic[fi['product_id']]
            if fi['is_product'] == '1':
                product['product_image'] = f
            else:
                product['model_images'].append(f)
                
    return dic
            
def select_model(l_product, product_indices, n_pair):
    '''
    select model from product
    
    l_product: list of product
    product_indices: selected products index
    n_pair: number of model per product
    
    returns:
    list of (product_file, model_file)
    '''
    r = []
    
    for idx in product_indices:
        product = l_product[idx]
        models = product['model_images']
        model_perm = np.random.permutation(len(models))
        for model_idx in model_perm[:min(len(models), n_pair)]:
            r.append((product['product_image'], models[model_idx]))
            
    return r

def pair_files(dic_product, n_train, n_test, n_pair):
    '''
    pair product file name with model's
    
    dic_product: a dictionary of 
        {'product_id': '000012', 
         'product_image': 'PID000000_CLEAN1_IID000011',
         'model_images': ['PID000000_CLEAN0_IID000000', ......]}
    n_train: number of product to select for train set
    n_test: number of product to select for test set
    
    returns:
    l_train: list of (product_file, model_file), note that the length may be less than n_pair * n_train
    l_test: list of (product_file, model_file)
    '''
    l_product = list(dic_product.values())
    product_perm = np.random.permutation(len(l_product))
    
    l_train = select_model(l_product, product_perm[:n_train], n_pair)
    l_test = select_model(l_product, product_perm[n_train:n_train + n_test], n_pair)
    
    return l_train, l_test

def combine_file_name(product, model):
    '''
    generate paired file name, in format like PID000292_CLEAN1_IID004184_IID004176.jpg
    
    product: file name of product, like PID000292_CLEAN1_IID004184.jpg
    model: file name of model, like PID000292_CLEAN0_IID004176.jpg
    
    returns: 
        file name of combined, like PID000292_CLEAN1_IID004184_IID004176.jpg
    '''
    last = len(product)
    return product[:last-4] + model[last-14:]

def resize_and_padding(image, desired_size = 64):
    """
    resize image to desired size
    keep it's ratio by padding it
    
    Arguments:
        image -- the image to resize
        desired_size -- the desire size of image, scalar
    
    Returns:
        the resized and padded image
    """
    old_size = image.shape[:2]  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    image_resized = resize(image, new_size, anti_aliasing=True)
    
    padding_top = (desired_size - new_size[0]) // 2
    padding_bottom = desired_size - new_size[0] - padding_top
    padding_left = (desired_size - new_size[1]) // 2
    padding_right = desired_size - new_size[1] - padding_left

    #if np.sum(image_resized >= 245.0/255.0) / image_resized.size > 0.65: 
    #    # most of the image is white
    image_padded = pad(image_resized, [(padding_top, padding_bottom), \
                                    (padding_left, padding_right), \
                                    (0, 0)], mode='constant', constant_values=1)
    #else:
    #    image_padded = pad(image_resized, [(padding_top, padding_bottom), \
    #                                   (padding_left, padding_right), \
    #                                    (0, 0)], mode='edge')
    
    assert(image_padded.shape[0] == desired_size)
    assert(image_padded.shape[1] == desired_size)
    return img_as_ubyte(image_padded)

def fix_channel(image):
    """
    fix number of channels to 3
    
    Arguments:
        image -- image to be processed
        
    Returns:
        fixed image
    """
    if len(image.shape) == 2:  # 1 channel image
        image_rgb = color.gray2rgb(image)
        return image_rgb
    elif len(image.shape) == 3 and image.shape[2] == 4:  # 4 channels image ie png
        image_reduced = image[:, :, :3]
        return image_reduced
    elif len(image.shape) == 3 and image.shape[2] == 3:  # normla image
        return image
    else:
        assert(False)
        
def stack_images(img1, img2):
    '''
    stack 2 image in same size horizontally
    
    img1: image in np array
    img2: image in np array
    
    returns:
        image in np array
    '''
    assert(img1.shape == img2.shape)
    img = np.hstack([img1, img2])
    return img

def generate_paired(opt):
    '''
    read lookbook images, select some of them, generate paired images and save
    
    opt : Options
    
    returns:
        None
    '''
    create_target_path(opt)
    
    # pair files
    files = get_lookbook_files(opt)
    dic_product = read_lookbook(files)
    train_pairs, test_pairs = pair_files(dic_product, opt.n_train, opt.n_test, opt.n_pair)
    
    # combine images
    for pairs, division in [(train_pairs, 'train'), (test_pairs, 'test')]:
        src_path = opt.src_folder + 'data/'
        target_path = opt.job_target_folder + 'image/' + division + '/'
        for product, model in pairs:
            img_product = io.imread(src_path + product)
            img_product = fix_channel(img_product)
            img_product = resize_and_padding(img_product, opt.image_size)
            
            img_model = io.imread(src_path + model)
            img_model = fix_channel(img_model)
            img_model = resize_and_padding(img_model, opt.image_size)
            
            img = stack_images(img_product, img_model)
            file = target_path + combine_file_name(product, model)
            io.imsave(file, img)
        