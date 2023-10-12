import os
import pandas as pd
import numpy as np
from common.config import Config


class PreClinical:

    def pre_process(self, df_clinical):
        # Censoring 9 -> 0
        df_clinical.loc[(df_clinical['Deadstatus.event'] == 9, 'Deadstatus.event')] = 0
        df_clinical['Overall.stage'] = df_clinical['Overall.stage'].str.upper()
        # Overall.stage 4 and 7
        # Overall.stage 값 변경

        # categorical data
        col_list = df_clinical.columns.tolist()

        # Remove Mcode and Mcode.description
        if 'Mcode' in col_list:
            df_clinical.drop(['Mcode'], axis='columns', inplace=True)
        if 'Mcode.description' in col_list:
            df_clinical.drop(['Mcode.description'], axis='columns', inplace=True)
        if "Histology" in col_list:
            df_clinical.drop(['Histology'], axis='columns', inplace=True)

        # Survival Label 추가
        year = 5  # 5년으로 나누기
        df_clinical['{}year_survival'.format(year)] = 0
        df_clinical.loc[(df_clinical['Survival.time'] <= year * 365, '{}year_survival'.format(year))] = 1  # alive: 0, dead: 1

        df_clinical_dummy = pd.get_dummies(df_clinical, columns=['gender', 'Overall.stage', 'Clinical.T.Stage',
                                                                 'Clinical.N.stage', 'Clinical.M.stage', 'Smoking.status'])

        output_path = f"./preprocess/output/{Config.img_shape}"

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        df_clinical_dummy.to_excel(f"{output_path}/clinical_pre.xlsx", index=False)
        print('>> [MESSAGE] SUCCESS: CLINICAL 전처리 완료')
        return df_clinical_dummy
