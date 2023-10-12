import os
import argparse
import pandas as pd
from common.config import Config
from preprocess.PreClinical import PreClinical
from preprocess.Gen3dImage import Gen3dImage
from preprocess.Normal3dImage import Normal3dImage
from preprocess.SplitData import split_censored
# from evaluation.Service_SC_Clinical_CT_PET import service_classify


def pre_process():
    ###############
    # 1.Data Load #
    ###############
    # Clinical, CT, PET 데이터셋 디렉토리 연결
    df_clinical = pd.read_excel("../data/LC_NSCLC_CLINICAL_n=2687.xlsx")
    df_clinical = df_clinical.copy()[:50]
    ct_list = os.listdir("../data/ct")
    pet_list = os.listdir("../data/pet")
    print('■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■')
    print('■□ 1.Number of Patients')
    print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
    print(f"■■ [CLINICAL 환자수] → {len(df_clinical)}명")
    print(f"■■ [CT 환자수] → {len(ct_list)}명")
    print(f"■■ [PET 환자수] → {len(pet_list)}명")

    # 경고메세지 출력
    if not (len(df_clinical) == len(ct_list) == len(pet_list)):
        print('>> [MESSAGE] ERROR: 멀티모달데이터셋 환자수가 같지 않습니다.')

    ###################
    # 2.PrePreprocess #
    ###################
    # 2-1. Clinical 데이터 전처리
    pc = PreClinical()
    df_clinical = pc.pre_process(df_clinical)

    # 2-2. 3D이미지 생성 및 reshape
    # ex) 50x128x128x1 → dim x width x height x channel
    print('■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■')
    print('■□ 2.Generate 3D Image')
    gi = Gen3dImage()

    # 환자 목록 생성
    patient_list = list(df_clinical['PatientID'])  # 임상데이터셋에서 환자ID 목록 추출

    # DICOM(CT, PET) 이미지 전처리
    pixel_size = Config.get_pixel_szie()  # → IMAGE WIDTH & HEIGHT
    dim = Config.get_dim()  # → IMAGE DIMENSION

    config = (pixel_size, dim)
    gi.gen_3d_image('CT', "../data/ct", patient_list)
    gi.gen_3d_image('PET', "../data/pet", patient_list)

    #####################
    ## 3.Normalization ##
    #####################
    print('■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■')
    print('■□ 3.Normalization 3D Image')
    ni = Normal3dImage()

    # 0~255 수치값으로 정규화
    ni.normal_3d_image('CT')
    ni.normal_3d_image('PET')

    ##########################
    ## 4.Train/Test 셋 분할 ##
    ##########################

    ## TRAIN, TEST → 8:1:1 비율로 Split
    ## : Censored 데이터 고려해서 (0, 1)에서 8:1:1 비율로 층화추출
    split_censored()

    # 5. Binary Survival Classification
    # service_classify(args.MODEL_PATH)


def main():
    # p = argparse.ArgumentParser()
    #
    # p.add_argument('-model', '--MODEL_PATH', default='')         # Survival Classification Model 로드
    # p.add_argument('-clinical', '--CLINICAL_PATH', default='')   # CLINICAL excel path
    # p.add_argument('-ct', '--CT_PATH', default='')               # CT dir path
    # p.add_argument('-pet', '--PET_PATH', default='')             # PET dir path
    #
    # args = p.parse_args()
    # CLINICAL, CT, PET 전처리
    pre_process()


if __name__ == '__main__':
    main()
