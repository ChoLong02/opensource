import os
import sys
import cv2
import numpy as np
import pydicom
import scipy.ndimage
from common.config import Config
# np.set_printoptions(threshold=sys.maxsize)


class Gen3dImage:

    def gen_3d_image(self, img_type, dir_name, patient_list):
        pixel_size = Config.get_pixel_szie()  # → IMAGE WIDTH & HEIGHT
        dim = Config.get_dim()  # → IMAGE DIMENSION

        # 이미지 전처리
        img3d_list = []
        for i, patient_id in enumerate(patient_list):
            # 이미지 불러오기
            print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
            print(f'■■ Preprocess: Start → {patient_id} 환자({img_type}-{i+1})')
            # 해당 환자의 250~450장의 전신 slice 이미지 번호를 정렬(ex: IM01, IM02, IM03 ... IM450)

            if img_type == 'CT':
                patient_id = 'CT_' + patient_id

            dcm_imgs = sorted(os.listdir(dir_name + '/' + patient_id), key=lambda img_name: int(img_name[2:]))

            # dicom 이미지 로드
            slices = []
            for img in dcm_imgs:
                slices.append(pydicom.dcmread(dir_name + '/' + patient_id + '/' + img))

            # 로드 된 dicom 이미지 정렬
            slices = sorted(slices, key=lambda s: s.SliceLocation)

            row, columns = slices[0].pixel_array.shape
            print('■■ [Original 3D Shape] → ({}, {}, {})'.format(len(slices), row, columns))


            # 3D 이미지 생성
            img_shape = list([pixel_size, pixel_size])  # → 3D 이미지 width x height 설정
            img_shape.insert(0, len(slices))
            img3d = np.zeros(img_shape)

            for j, s in enumerate(slices):

                if row != pixel_size :
                    img2d = s.pixel_array
                    # print('[resize before] →', img2d.shape)
                    resized_img = cv2.resize(img2d, dsize=(pixel_size, pixel_size))
                    # print('[resize after] →', resized_img.shape)

                    img3d[j, :, :] = resized_img
                elif img_type == 'PET':
                    img3d[j, :, :] = s.pixel_array
            # print('■■ [Input 3D Shape] → ', img3d.shape)

            # → 3D 이미지 차원 줄이기
            # 배경 채우기
            CT_background = -2000.0
            PET_background = 0
            if dim > len(slices):
                if img_type == 'CT':
                    background_val = CT_background
                else:
                    background_val = PET_background

                padding_img3d = np.full([(dim - len(slices)), pixel_size, pixel_size], background_val, dtype=np.float32)
                img3d = np.concatenate((img3d, padding_img3d), axis=0)
            else:
                x = dim / len(slices)
                real_resize_factor = np.array([x, 1, 1], dtype=np.float32)

                img3d = scipy.ndimage.interpolation.zoom(img3d, real_resize_factor)

            img3d = img3d.astype('float32')
            # img3d = abs(img3d) / 255 # → PET의 음수값을 절대값으로 변환
            img3d = np.reshape(img3d, (dim, pixel_size, pixel_size, 1))
            print('■■ [Output 3D Shape] → ', img3d.shape)
            img3d_list.append(img3d)

            # if i == 30:
            #      break

        # 3D Numpy 이미지 저장하기
        output_path = f'./preprocess/output/{Config.img_shape}'

        # 이미지를 저장하는 폴더가 없으면 폴더 생성
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        np.save(f'{output_path}/{img_type}_{Config.img_shape}', img3d_list)
        print('>> [MESSAGE] SUCCESS: {} → 3D이미지(Numpy) 생성'.format(img_type))
