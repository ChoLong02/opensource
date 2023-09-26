import os
import numpy as np
import random as rd
import pandas as pd
from common.config import Config


def split_censored():
    df_clinical = pd.read_excel('./preprocess/output/{}/CLINICAL_pre.xlsx'.format(Config.img_shape))
    normal_ct = np.load('./preprocess/output/{}/CT_{}_normal.npy'.format(Config.img_shape, Config.img_shape))
    normal_pet = np.load('./preprocess/output/{}/PET_{}_normal.npy'.format(Config.img_shape, Config.img_shape))

    print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
    print('■■ Split TRAIN/TEST: Start')

    # TRAIN, TEST : 7, 3 비율로 자르는 Index Num 생성
    seed_ = 12345
    rd.seed(seed_)

    # alive = np.where(df_clinical['Deadstatus.event']==0)[0]
    # dead = np.where(df_clinical['Deadstatus.event']==1)[0]
    alive = np.where(df_clinical['5year_survival'] == 0)[0]
    dead = np.where(df_clinical['5year_survival']== 1)[0]

    rd.shuffle(alive)
    rd.shuffle(dead)

    p = [0, 0.8, 0.1, 0.11]
    p = np.cumsum(np.array(p))

    split_idx = []
    for i in range(len(p)-1) :
        split_num = np.hstack((alive[int(len(alive)*p[i]):int(len(alive)*p[i+1])],dead[int(len(dead)*p[i]):int(len(dead)*p[i+1])]))
        rd.shuffle(split_num)
        split_idx.append(split_num)

    train_num = split_idx[0]
    valid_num = split_idx[1]
    test_num = split_idx[2]

    print('■■ TRAIN({}) + VALID({}) + TEST({}) = {}'.format(len(train_num), len(test_num), len(valid_num), (len(train_num) + len(valid_num) + len(test_num))))

    # CLINICAL 임상데이터 Split
    df_train = df_clinical.loc[train_num]
    df_valid = df_clinical.loc[valid_num]
    df_test = df_clinical.loc[test_num]

    df_train.to_excel('./preprocess/output/{}/TRAIN_CLINICAL_pre.xlsx'.format(Config.img_shape), index=False)
    df_valid.to_excel('./preprocess/output/{}/VALID_CLINICAL_pre.xlsx'.format(Config.img_shape), index=False)
    df_test.to_excel('./preprocess/output/{}/TEST_CLINICAL_pre.xlsx'.format(Config.img_shape), index=False)

    # DICOM CT&PET 이미지 Split
    pixel_size = Config.get_pixel_szie()  # → IMAGE WIDTH & HEIGHT
    dim = Config.get_dim()  # → IMAGE DIMENSION
    train_ct = normal_ct[split_idx[0]]
    valid_ct = normal_ct[split_idx[1]]
    test_ct = normal_ct[split_idx[2]]

    train_pet = normal_pet[split_idx[0]]
    valid_pet = normal_pet[split_idx[1]]
    test_pet = normal_pet[split_idx[2]]

    output_path = '../output/{}'.format(Config.img_shape)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    np.save('./preprocess/output/{}/TRAIN_CT_{}_normal'.format(Config.img_shape, Config.img_shape), train_ct)
    np.save('./preprocess/output/{}/VALID_CT_{}_normal'.format(Config.img_shape, Config.img_shape), valid_ct)
    np.save('./preprocess/output/{}/TEST_CT_{}_normal'.format(Config.img_shape, Config.img_shape), test_ct)
    np.save('./preprocess/output/{}/TRAIN_PET_{}_normal'.format(Config.img_shape, Config.img_shape), train_pet)
    np.save('./preprocess/output/{}/VALID_PET_{}_normal'.format(Config.img_shape, Config.img_shape), valid_pet)
    np.save('./preprocess/output/{}/TEST_PET_{}_normal'.format(Config.img_shape, Config.img_shape), test_pet)

    return split_idx
