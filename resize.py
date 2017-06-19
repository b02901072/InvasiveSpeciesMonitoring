#!/usr/bin/env python3

'''
    resize.py: resize all raw face images to a certain size
'''

import skimage
import skimage.io
import skimage.transform
import scipy.misc
import glob
import os

def resize_images(img_dirname, size):
    '''
        Resize all raw face images to a certain size
    '''
    new_img_dirname = img_dirname + '_' + str(size[0]) + 'x' + str(size[1])
    print('Source images directory: %s' % img_dirname)
    print('Target images directory: %s' % new_img_dirname)
    if not os.path.exists(new_img_dirname):
        os.makedirs(new_img_dirname)
    print('Resize images size to %s' % str(size))
    count = 0
    for img_filename in glob.glob(os.path.join(img_dirname, '*.jpg')):
        img_basename = os.path.basename(img_filename)
        img_id = int(img_basename[:-4])
        img = skimage.io.imread(img_filename)
        new_img_filename = os.path.join(new_img_dirname, str(img_id)+'.jpg')
 
        img_resized = skimage.transform.resize(img, size, mode='constant')

        count += 1
        print('Resized images count: %s' % count, end='\r')
        
        scipy.misc.imsave(new_img_filename, img_resized)
    print()
    print('All images resized')
    
if __name__ == '__main__':
    resize_images('data/train', (128, 128))
    resize_images('data/test', (128, 128))

