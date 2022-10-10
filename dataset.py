#-*_ coding:utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def RandomCrop(image, output_size):
    rows,cols = image.shape[:2]
    row_point = np.random.randint(0, rows - output_size)
    col_point = np.random.randint(0, cols - output_size)
    # dst = image[row_point[0]:row_point[0] + output_size, col_point[0]:col_point[0] + output_size]
    dst = image[row_point:row_point + output_size, col_point:col_point + output_size]
    return dst

# Train
def get_dataframe(k_fold, data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'train.csv'))

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}train', x))  # f'{x}.jpg'
    # df_train['org_img_name'] = df_train['image_name'].apply(lambda x: (x.split('_')[0]))

    # 원본데이터=0, 외부데이터=1
    df_train['is_ext'] = 0

    '''
    ####################################################
    교차 검증 구현 (k-fold cross-validation)
    ####################################################
    '''
    # 교차 검증을 위해 이미지 리스트들에 분리된 번호를 매김
    img_ids = len(df_train['img_id'].unique())
    print(f'Original dataset의 이미지수 : {img_ids}')

    # 데이터 인덱스 : fold 번호. (fold)번 분할뭉치로 간다
    # train.py input arg에서 k-fold를 수정해줘야함 (default:5)
    # print(f'Dataset: {k_fold}-fold cross-validation')
    # img_id2fold = {i: i % k_fold for i in range(img_ids)}
    # df_train['fold'] = df_train['img_id'].map(img_id2fold)


    # test data (학습이랑 똑같게 함)
    # df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    # df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'

    return df_train#, df_test

# Train
# Gray-Down-Crop(512)-까지 된 상태
class resamplingDataset_orgimg(Dataset): # train dataset 동적으로 만드는 class
    def __init__(self, csv, mode, image_size=256, transform=None):

        self.csv = pd.concat([csv], ignore_index=True)
        self.mode = mode  # train / valid
        self.transform = transform
        self.image_size = image_size

        # self.r_test_delta = 45
        # self.s_test_delta = 0.05

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        image = cv2.imread(self.csv.iloc[index].filepath, cv2.IMREAD_GRAYSCALE)
        rows, cols = image.shape[:2]
        image_center = (cols // 2, rows // 2)
        ################################################

        while(True):
            scale = round(int(np.random.uniform(low=10, high=40, size=(1,))) * 0.05, 2)
            rot = int(np.random.uniform(low=0, high=46, size=(1,)))

            if(scale==1.0 and rot==0):
                continue
            else:
                break

        ################################################
        matrix = cv2.getRotationMatrix2D(image_center, float(rot), float(scale))
        dst = cv2.warpAffine(image, matrix, (cols, rows), cv2.INTER_LINEAR)
        dst = dst[rows // 2 - 128:rows // 2 + 128, cols // 2 - 128:cols // 2 + 128]
        ################################################
        # center crop

        # Matrix
        target_list = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]  # scaling factor, scaling시에만 적용된다
        factor = [scale, rot]
        # Classification & Regression
        # target_list = [scal_factor, rot_factor]

        res = self.transform(image=dst)
        image = res['image'].astype(np.float32)
        #########################################

        image = np.expand_dims(image, axis=0)
        ################
        # image = cv2.Laplacian(image, cv2.CV_32F, -1)  ###############################################################################################################################
        # image = np.transpose(image, (2, 0, 1))  # GrayScale
        ################
        data = torch.tensor(image).float()

        return data, torch.tensor(target_list).float(), torch.tensor(factor).float()
#########################################

#########################################
# Validation
def get_dataframe_val(data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'valid_scaling.csv')) # scaling, rotation, all
    df_train2 = pd.read_csv(os.path.join(data_dir, data_folder, 'valid_rot.csv'))
    df_train3 = pd.read_csv(os.path.join(data_dir, data_folder, 'valid_all.csv'))
    # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # df_train = df_train[df_train['인덱스 이름'] != -1].reset_index(drop=True)

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    # df_train['image_name'] = df_train['image_name'] + '_' + df_train['patch'].astype(str)
    # df_train['image_name'] = df_train['image_name'].apply(lambda x: x+'.png')
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}valid_scaling', f'{x}.png'))  # f'{x}.jpg'

    # df_train2['image_name'] = df_train2['image_name'] + '_' + df_train2['patch'].astype(str)
    # df_train2['image_name'] = df_train2['image_name'].apply(lambda x: x + '.png')
    df_train2['filepath'] = df_train2['image_name'].apply(
        lambda x: os.path.join(data_dir, f'{data_folder}valid_rot', f'{x}.png'))  # f'{x}.jpg'

    df_train3['filepath'] = df_train3['image_name'].apply(
        lambda x: os.path.join(data_dir, f'{data_folder}valid_all', f'{x}.png'))  # f'{x}.jpg'
    # 원본데이터=0, 외부데이터=1


    # #####
    # s_factor2class = {round(s * 0.05, 2): idx for idx, s in enumerate(range(10, 40))}  # 30가지
    # r_factor2class = {s: idx for idx, s in enumerate(range(0, 45))}  # 46가지 ==> 1379
    # df_train3['s_class'] = df_train3['scaling'].map(s_factor2class)
    # df_train3['r_class'] = df_train3['rotation'].map(r_factor2class)
    #
    # df_train3['class'] = (df_train3['s_class'].astype('str')).str.cat((df_train3['r_class']).astype('str'), sep="_")
    # sr_factor2class = {s: idx for idx, s in enumerate(df_train3['class'].unique())}  # 1340가지
    # df_train3['class2'] = df_train3['class'].map(sr_factor2class)
    #
    # ####

    return df_train, df_train2, df_train3 #mani_class, df_train2, mani_class2


# Validation - Classification, Regression, Matrix
class resamplingDataset_val(Dataset):
    def __init__(self, csv, mode, image_size=256, transform=None):
        self.csv = csv
        self.mode = mode # train / valid
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        image = cv2.imread(self.csv.iloc[index].filepath, cv2.IMREAD_GRAYSCALE)

        target_list = [np.round(self.csv.iloc[index].p1,2), np.round(self.csv.iloc[index].p2,2), np.round(self.csv.iloc[index].p4,2), np.round(self.csv.iloc[index].p5,2)]   #scaling factor, scaling시에만 적용된다
        target_factor = [np.round(self.csv.iloc[index].scaling,2), np.round(self.csv.iloc[index].rotation,1)]

        # albumentation 적용
        res = self.transform(image=image)
        image = res['image'].astype(np.float32)
        #########################################
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # 흑백 이미지 변환 후 차원 변경 [1, 1024, 1024]
        image = np.expand_dims(image, axis=0)
        #########################################
        ################
        # image = cv2.Laplacian(image, cv2.CV_32F, -1) ###############################################################################################################################
        # image = np.transpose(image, (2, 0, 1))  # GrayScale
        ################

        # 학습용 데이터 리턴
        data = torch.tensor(image).float()

        return data, torch.tensor(target_list).float(), torch.tensor(target_factor).float()


def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        # albumentations.RandomBrightness(limit=0.1, p=0.75),
        # albumentations.RandomContrast(limit=0.1, p=0.75),
        # albumentations.CLAHE(clip_limit=2.0, p=0.3),
        # albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        # albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=1, p=0.7),
        # albumentations.RandomGamma(gamma_limit=(80, 120), eps=None,always_apply=False, p=0.5),
        albumentations.Normalize(mean=(0.5419184366861979),std=(0.14091745018959045))
        # shift
    ])

    transforms_val = albumentations.Compose([
        albumentations.Normalize(mean=(0.5419184366861979),std=(0.14091745018959045))
    ])

    return transforms_train, transforms_val

def get_meta_data_stoneproject(df_train, df_test):
    '''
    ####################################################
                        안씀
    ####################################################
    '''

    return 0,0,0,0


