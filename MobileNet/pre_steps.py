import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
import random
from sklearn.model_selection import train_test_split


def save_data_in_folder(random_seed=1):
    base_skin_dir = os.path.join('.', 'input')

    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

    # random.seed(random_seed)
    data_num = tile_df.shape[0]
    total_index = list(range(data_num))
    train_index, test_index = train_test_split(total_index, test_size=0.1, random_state=random_seed)

    train_test_flag = []
    for i in range(data_num):
        if i in train_index:
            train_test_flag.append('train')
        else:
            train_test_flag.append('test')
    tile_df['flag'] = train_test_flag
    #tile_df['new_path'] = tile_df['flag'].str.cat(tile_df[['dx','image_id']], sep = '/')
    tile_df['new_path'] = tile_df[['flag','dx', 'image_id']].apply(lambda x: './input/{}/{}/{}.jpg'.format(x[0], x[1], x[2]), axis=1)
    #print(tile_df[['path', 'new_path']])
    # code to distribute the files into different sub-folders.
    print(list(tile_df))
    for dx in pd.unique(tile_df['dx']):
        os.makedirs('./input/train/{}'.format(dx))
        os.makedirs('./input/test/{}'.format(dx))
    for index, row in tile_df.iterrows():
        os.rename(row['path'], row['new_path'])
        pass

#save_data_in_folder()



def generate_gaussian_noise():
    train_num = {'akiec': 299,
                 'bcc': 467,
                 'bkl': 977,
                 'df': 102,
                 'mel': 1002,
                 'nv': 6045,
                 'vasc': 128}

    import numpy as np
    import random
    import cv2
    def sp_noise(image):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    file_dic = {}
    for class_dir in os.listdir('./input/train'):
        target_num = 6045 - train_num[class_dir]
        file_dic[class_dir] = []
        if train_num == 0:
            continue
        class_path = os.path.join('./input/train/', class_dir)
        target_dir = os.path.join('/data/qinqingliu/2018Fall/MobileNet/input_noisy/train/', class_dir)
        os.makedirs(target_dir)

        for file in os.listdir(class_path):
            image = cv2.imread(os.path.join(class_path, file))
            cv2.imwrite(
                '/data/qinqingliu/2018Fall/MobileNet/input_noisy/train/{}/{}'.format(class_dir, file), image)
            file_dic[class_dir].append(file)

        index = 0
        while index < target_num:
            file = random.choice(file_dic[class_dir])
            name = file.split('.')[0]
            image = cv2.imread(os.path.join(class_path, file))
            noise_img = sp_noise(image)
            cv2.imwrite('/data/qinqingliu/2018Fall/MobileNet/input_noisy/train/{}/{}_sp_noise_{}.jpg'.format(class_dir, name, index), noise_img)
            index += 1
            #print('Finished {}'.format(file))


    #image = cv2.imread('image.jpg', 0)  # Only for grayscale image
    #noise_img = sp_noise(image)
    #des_pth = '/data/qinqingliu/2018Fall/MobileNet/input_noisy/train'
    #cv2.imwrite('sp_noise.jpg', noise_img)

#generate_gaussian_noise()



