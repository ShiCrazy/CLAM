

import torch
import os
import time

def load_model(file_path):
    try:
        model_state_dict = torch.load(file_path)
        return model_state_dict
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def load_all_models_in_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith('.pt'):
                file_path = os.path.join(root, file_name)
                s_time = time.time()
                data = load_model(file_path)
                e_time = time.time()
                print('elapsed time:', e_time-s_time)
                if data is not None:
                    print(data)
                    print(data.shape)
                    # 检查数据是否为字典，并遍历字典中的每个张量
                    if isinstance(data, dict):
                        for key, tensor in data.items():
                            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0 and tensor.size(0) < 10:
                                print(f"文件 {file_name} 包含长度小于10的张量，键名为 {key}。")
                    # 检查数据是否为单个张量
                    elif isinstance(data, torch.Tensor) and data.dim() > 0 and data.size(0) < 20:
                        print(f"文件 {file_name} 包含长度小于20的张量，dim=0的长度为 {data.size(0)}。")
                else:
                    print(f"文件 {file_name} 读取失败.")

directory_path = '/wz-pstor02/Shigb_CLAM_FGFR/gamma/FEATURES/Ori_fgfr_icc_2016.7.28-2023.6.28/pt_files'
directory_path = '/wz-pstor02/Shigb_CLAM_FGFR/clam_ostu_fgfr_beta/FEATURES/Ori_fgfr2_icc_2016.7.28-2023.6.28/pt_files'
directory_path = '/wz-pstor02/Shigb_CLAM_FGFR/gamma/FEATURES/11'

load_all_models_in_directory(directory_path)


