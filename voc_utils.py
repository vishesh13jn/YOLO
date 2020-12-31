import skimage
from skimage import io
import pandas as pd
from bs4 import BeautifulSoup
import os
from more_itertools import unique_everseen
import matplotlib.pyplot as plt
import numpy as np

source_directory = '/home/dani/Files/Data/VOC2012/'
directory_image = os.path.join(source_directory, 'JPEGImages/')
dir_ann = os.path.join(source_directory, 'Annotations/')
dir_set = os.path.join(source_directory, 'ImageSets', 'Main')

def image_list():         #getting list of images
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

def description_file_from_img(name_of_image):        #setting path for description file of images
    return os.path.join(dir_ann, name_of_image) + '.xml'

def images_from_diff_category(name_of_category, dataset):
    name_of_file = os.path.join(dir_set, name_of_category + "_" + dataset + ".txt")
    df = pd.read_csv(
        name_of_file,
        delim_whitespace=True,
        header=None,
        names=['name_of_file', 'true'])
    return df

def images_from_diff_category_as_list(name_of_category, dataset):    #images of different categories in form of list
    df = images_from_diff_category(name_of_category, dataset)
    df = df[df['true'] == 1]
    return df['name_of_file'].values

def load_imgs(name_of_file_imagefiles):                #loading images
    return np.array([load_img(fname) for fname in name_of_file_imagefiles])

def get_masks(name_of_category, data_type, type_of_mask=None):        #getting masks
    if type_of_mask is None:
        raise ValueError('Must specify type_of_mask')
    df = data_loading(name_of_category, data_type=data_type)

    masks = []
    prev_url = ""
    blank_img = None
    for row_num, entry in df.iterrows():
        url_of_image = os.path.join(directory_image, entry['fname'])
        if url_of_image != prev_url:
            if blank_img is not None:
                value_maximum = blank_img.max()
                if value_maximum > 0:
                    value_minimum = blank_img.min()
                    blank_img -= value_minimum
                    blank_img /= value_maximum
                masks.append(blank_img)
            prev_url = url_of_image
            img = load_img(url_of_image)
            blank_img = np.zeros((img.shape[0], img.shape[1], 1))
        bbox = [entry['min_x'], entry['min_y'], entry['max_x'], entry['max_y']]
        if type_of_mask == 'bounding box1':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
        elif type_of_mask == 'bounding box2':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 1.0
        else:
            raise ValueError('It is not a valid mask type')
    value_maximum = blank_img.max()
    if value_maximum > 0:
        value_minimum = blank_img.min()
        blank_img -= value_minimum
        blank_img /= value_maximum
    masks.append(blank_img)
    return np.array(masks)

def display_image_and_mask(img, mask):           #display the images and mask
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(mask)
    ax2.set_title('Mask')
    plt.show(block=False)

def load_description(name_of_file_imagefile):        #load description
    xml = ""
    with open(description_file_from_img(name_of_file_imagefile)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)

def get_all_obj_and_box(objname, img_set):            #get all objects and box
    img_list = images_from_diff_category_as_list(objname, img_set)

    for img in img_list:
        description = load_description(img)

def load_img(name_of_file_imagefile):          #load image
    name_of_file_imagefile = os.path.join(directory_image, name_of_file_imagefile + '.jpg')
    img = skimage.img_as_float(io.imread(
        name_of_file_imagefile)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def load_data_multilabel(data_type=None):          #load data of multilabel
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    name_of_file = os.path.join(dir_set, data_type + ".txt")
    list_of_cat = image_list()
    df = pd.read_csv(
        name_of_file,
        delim_whitespace=True,
        header=None,
        names=['name_of_file'])
    for name_of_category in list_of_cat:
        df[name_of_category] = 0
    for info in df.itertuples():
        index = info[0]
        fname = info[1]
        annotation = load_description(fname)
        objs = annotation.findAll('object')
        for obj in objs:
            names_objects = obj.findChildren('name')
            for name_tag in names_objects:
                tag_name = str(name_tag.contents[0])
                if tag_name in list_of_cat:
                    df.at[index, tag_name] = 1
    return df

def data_loading(category, data_type=None):       #loading data
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    to_find = category
    name_of_file = os.path.join(source_directory, 'csvs/') + \
        data_type + '_' + \
        category + '.csv'
    if os.path.isfile(name_of_file):
        return pd.read_csv(name_of_file)
    else:
        train_img_list = images_from_diff_category_as_list(to_find, data_type)
        data = []
        for item in train_img_list:
            annotation = load_description(item)
            objs = annotation.findAll('object')
            for obj in objs:
                names_objects = obj.findChildren('name')
                for name_tag in names_objects:
                    if str(name_tag.contents[0]) == category:
                        fname = annotation.findChild('name_of_file').contents[0]
                        bbox = obj.findChildren('bndbox')[0]
                        minimum_x = int(bbox.findChildren('minimum_x')[0].contents[0])
                        minimum_y = int(bbox.findChildren('minimum_y')[0].contents[0])
                        maximum_x = int(bbox.findChildren('maximum_x')[0].contents[0])
                        maximum_y = int(bbox.findChildren('maximum_y')[0].contents[0])
                        data.append([fname, minimum_x, minimum_y, maximum_x, maximum_y])
        df = pd.DataFrame(
            data, columns=['fname', 'minimum_x', 'minimum_y', 'maximum_x', 'maximum_y'])
        df.to_csv(name_of_file)
        return df

def list_url_of_image(category, data_type=None):            #list the url of images
    df = data_loading(category, data_type=data_type)
    image_url_list = list(
        unique_everseen(list(directory_image + df['fname'])))
    return image_url_list

def get_imgs(name_of_category, data_type=None):            #get the images
    image_url_list = list_url_of_image(name_of_category, data_type=data_type)
    imgs = []
    for url in image_url_list:
        imgs.append(load_img(url))
    return np.array(imgs)

def display_img_and_masks(                                  #display images and masks
        img, true_mask, predicted_mask, block=False):
    m_predicted_color = predicted_mask.reshape(
        predicted_mask.shape[0], predicted_mask.shape[1])
    m_true_color = true_mask.reshape(
        true_mask.shape[0], true_mask.shape[1])

    plt.figure(1)
    plt.clf()
    plt.axis('off')
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, num=1)
    
    ax1.get_xaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])

    ax1.imshow(img)
    ax2.imshow(m_true_color)
    ax3.imshow(m_predicted_color)
    plt.draw()
    plt.show(block=block)
    
def category_name_to_id_cat(name_of_category):
    list_of_cat = image_list()
    cat_id_dict = dict(zip(list_of_cat, range(len(list_of_cat))))
    return cat_id_dict[name_of_category]