from google.colab import drive
drive.mount("/content/drive")

import pandas as pd
import glob
import os

tabular_csv_path = '/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/data/package-plco-1514/Lung/lung_data_mar22_d032222.csv'
images_csv_path = '/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/data/package-plcoi-1517-file/Lung/Standard 25K Linkage (2021)/lung_xryimg25_data_mar22_d032222.csv'
image_database_path = '/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/data/package-plcoi-1517'

# Read tabular data - 154,887 samples
tabular_df = pd.read_csv(tabular_csv_path)
# Read imaging related data - 89,716 samples
images_df = pd.read_csv(images_csv_path)

print(tabular_df.shape, images_df.shape)

# Keep only samples included in the imaging data - 25,000 samples
tabular_df = tabular_df[tabular_df['selected_25k_2020'] == 1]
# Keep only samples at year 0 - 25,446 samples
images_df = images_df[images_df['study_yr'] == 0]

print(tabular_df.shape, images_df.shape)

# Keep only samples that have diagnosticable images - 23,803 samples
tabular_df = tabular_df[tabular_df['xry_result0'].isin([1, 2, 3])]
# Filter samples that do not have clinical staging details - 23,704 samples
tabular_df = tabular_df[tabular_df['lung_cancer'] == 0 | tabular_df['lung_clinstage'].isin([110., 120., 210., 220., 310., 320., 400.])]
# Drop duplicates on imaging related data by keeping the last one - 23,803 samples
images_df = images_df.drop_duplicates(subset=['plco_id'], keep='last')

# Merge dataframes - 23,694 samples
main_df = pd.merge(tabular_df, images_df, on='plco_id')

print(tabular_df.shape, images_df.shape, main_df.shape)
main_df.head()

# ID
feat_id = ['plco_id', 'image_file_name']
# Input features
feat_demographics = ['age', 'sex', 'race7', 'bmi_curr', 'bmi_20']
feat_history = ['cigpd_f', 'cig_years', 'rsmoker_f', 'lung_fh', 'asp', 'ibup']
feat_comorbidity = ['arthrit_f', 'bronchit_f', 'colon_comorbidity', 'diabetes_f', 'divertic_f', 'emphys_f',
                    'gallblad_f', 'hearta_f', 'hyperten_f', 'liver_comorbidity', 'osteopor_f', 'polyps_f', 'stroke_f']
# Output features
feat_output = ['lung_cancer', 'lung_clinstage']

# Filtered data (23694, 27)
main_df = main_df[feat_id + feat_demographics + feat_history + feat_comorbidity + feat_output]

print(main_df.shape)

# Adjust lung_clinstage - 0 control, 1 early lung cancer stage, 2 advanced stage - optional for multiple classes
main_df['lung_clinstage'] = main_df['lung_clinstage'].fillna(0)
main_df['lung_clinstage'] = main_df['lung_clinstage'].replace([110., 120., 210., 220.], 1)
main_df['lung_clinstage'] = main_df['lung_clinstage'].replace([310., 320., 400.], 2)

# Filter out missing values (12494, 28) - many controls were excluded due to missing rsmoker_f
main_df = main_df.dropna()

# Adjust data types
main_df[feat_history + feat_comorbidity + feat_output] = main_df[feat_history + feat_comorbidity + feat_output].astype('int32')
main_df[['age', 'sex', 'race7']] = main_df[['age', 'sex', 'race7']].astype('int32')
main_df[['bmi_curr', 'bmi_20']] = main_df[['bmi_curr', 'bmi_20']].astype('float32')

# Adjust values
main_df['sex'] = main_df['sex'] - 1                     # 0 for male and 1 for female
main_df['race7'] = main_df['race7'] - 1                 # starting from 0, to 6
main_df['lung_fh'] = main_df['lung_fh'].replace(9, 0)   # changed (few) possibly to yes

main_df.describe()

# Get list of all image paths
image_paths = glob.glob(os.path.join(image_database_path, '*', '*.jpg'))
path_df = pd.DataFrame(image_paths, columns=['path'])
path_df['image_file_name'] = path_df['path'].apply(lambda x: os.path.basename(x).split('.')[0])

# Link image_file_name with image_paths
main_df = pd.merge(main_df, path_df, on='image_file_name', how='left')

print(main_df.shape)

# Save
save_path = '/content/drive/MyDrive/Veritas AI/Veritas AI - Arjun/data'
main_df.to_csv(os.path.join(save_path, 'cleaned_data.csv'), index=False, encoding='utf-8')
