import subprocess
import argparse
import os
import shutil
import time
import re
import random
import cv2

script_dir = os.path.dirname(os.path.realpath(__file__))
new_folder_dir = os.path.dirname(script_dir)
openslide_bin_path = os.path.join(new_folder_dir, 'openslide-win64-20230414', 'bin')

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(os.path.abspath(openslide_bin_path)):
        import openslide
else:
    import openslide

# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
from utils.eval_utils import *
from PIL import Image

# other imports
import numpy as np
import csv
import pdb
import pandas as pd
import torch
import torch.nn as nn
from math import floor
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# """ delete_intermediate_files.py """ #
def remove_intermediate_files(slide_dir_path):
    slide_dir_path = slide_dir_path.rstrip('/')
    head, tail = os.path.split(slide_dir_path)
    intermedia_dir = ["./PATCHES", "./FEATURES", "./slides_name_csv", "./dataset_csv"]
    for i in intermedia_dir:
        if i == "./slides_name_csv":
            if os.path.exists(os.path.join(i, tail + "_slides_name.csv")):
                os.remove(os.path.join(i, tail + "_slides_name.csv"))
            else:
                pass
        elif i == "./dataset_csv":
            if os.path.exists(os.path.join(i, tail + ".csv")):
                os.remove(os.path.join(i, tail + ".csv"))
            else:
                pass
        else:
            if os.path.exists(os.path.join(i, tail)):
                shutil.rmtree(os.path.join(i, tail))
            else:
                pass


# “”“ create_patches_fp.py """ #
def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                  patch_size=256, step_size=256,
                  seg_params=None,
                  filter_params=None,
                  vis_params=None,
                  patch_params=None,
                  patch_level=0,
                  use_default_params=False,
                  seg=False, save_mask=True,
                  stitch=False,
                  patch=False, auto_skip=True, process_list=None):
    if seg_params is None:
        seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    if filter_params is None:
        filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    if vis_params is None:
        vis_params = {'vis_level': -1, 'line_thickness': 500}
    if patch_params is None:
        patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                          'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                          'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                          'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                          'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e9:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                         'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params, )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times


# """ create_slides_name_csv.py """ #
def create_slides_name_csv(path, name):
    slides = os.listdir(path)
    slide_id = [i.split(".h5")[0] for i in slides]
    frame = pd.DataFrame({'slide_id': slide_id})
    dir_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), "slides_name_csv")
    os.makedirs(dir_name, exist_ok=True)
    frame.to_csv(os.path.join(dir_name, name + "_slides_name.csv"), index=False)


# """ extract_features_fp.py """ #


def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    x, y = dataset[0]
    num_workers = 4 if os.name == 'posix' else 0
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


# """ generate_dataset_csv.py """ #
def generate_dataset_csv(svs_dir_path):
    svs_files_list = os.listdir(svs_dir_path)
    data = [[], [], []]
    for i in svs_files_list:
        data[0].append(i.rstrip('.svs'))
        data[1].append(i.rstrip('.svs'))
        data[2].append("negative")
    data = {'case_id': data[0], 'slide_id': data[1], 'label': data[2]}
    dframe = pd.DataFrame(data)
    return dframe


# """generate_eval_splits.csv""" #
def create_csv_files(input_file, output_dir, num_files=10):
    with open(input_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        slide_ids = [row[1] for row in reader]

    for i in range(num_files):
        output_file = os.path.join(output_dir, f"splits_{i}.csv")
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "train", "val", "test"])
            for idx, slide_id in enumerate(slide_ids):
                writer.writerow([idx, "", "", slide_id])


# """ eval.py """ #
def limited_float(value):
    value_float = float(value)
    if value_float < 0.0 or value_float > 1.0:
        raise argparse.ArgumentTypeError(f"输入值 {value} 不在允许的范围内 (0-1)")
    return value_float


# """eval_ensemble.py""" #
def return_k_files(dir_path):
    files = os.listdir(dir_path)
    files = [os.path.join(dir_path, file) for file in files if re.match(r"^fold_\d+\.csv$", file)]
    return files


def average_predictions(csv_files, threshold):
    threshold = float(threshold)

    data_frames = []

    for file in csv_files:
        data = pd.read_csv(file)
        data_frames.append(data)

    # 合并所有子模型的预测结果
    merged_data = pd.concat(data_frames)

    # 计算每个子模型的平均预测概率
    mean_predictions = merged_data.groupby("slide_id")[["p_0", "p_1"]].mean()

    # 添加"cutoff threshold"列
    mean_predictions["cutoff threshold"] = threshold

    # 添加 "Final Prediction Result" 列
    mean_predictions["Final Prediction Result"] = mean_predictions.apply(
        lambda row: "FGFR-ONCO" if row["p_1"] > row["cutoff threshold"] else "FGFR-WT",
        axis=1
    )

    # 重置索引
    final_data = mean_predictions.reset_index()

    return final_data


# """RUN_MODEL.py"""


def check_source_slides_dir(input_parameters):
    source_slides_dir_path = input_parameters.source_slides_dir
    for i in os.listdir(source_slides_dir_path):
        _, extension = os.path.splitext(i)
        if extension != ".svs":
            raise Exception("检测到其他格式的文件，请确保目录中只有svs格式后缀的病理图像！")


def check_slides_features_correspondence(input_parameters, intermediate_parameteres):
    input_slides_length = len(os.listdir(input_parameters.source_slides_dir))
    intermediate_features_length = len(os.listdir(os.path.join(intermediate_parameteres.features_save_dir, "h5_files")))
    if input_slides_length != intermediate_features_length:
        difference = input_slides_length - intermediate_features_length
        print(f"在输入病理图像中，有{difference}张病理图像无法被本程序所识别，故不产生对应的预测结果，建议重新获取对应的病理图像。"
              f"在训练该模型所使用的的至本数据库病理图像中，这种情况发生的概率大约为1%；在训练该模型所使用的TCGA数据库病理图像中，"
              f"这种情况发生的概率为0%")
        print()


def check_maximum_resolution(input_parameters):
    input_slides = os.listdir(input_parameters.source_slides_dir)
    obj_pow_20_list = []
    obj_pow_40_list = []
    obj_pow_other_list = []
    for i in input_slides:
        I = os.path.abspath(os.path.join(input_parameters.source_slides_dir, i))
        slide = openslide.OpenSlide(I)
        objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        if objective_power == 20:
            obj_pow_20_list.append(i)
        elif objective_power == 40:
            obj_pow_40_list.append(i)
        else:
            obj_pow_other_list.append(i)
    if len(obj_pow_20_list + obj_pow_other_list) > 0:
        print("尽管我们的程序支持处理各种最大物镜放大倍数的病理图像，但是请根据你选择的模型来选择你输入的病理图像，"
              "如果将最大放大倍数为40倍的病理图像输入20倍模型，结果会受到影响，反之若将最大放大倍数为20倍的病理图像输入40倍模型，结果也同样受到影响，请知晓")
        print("以下是最大物镜放大倍数为40倍的病理图像：")
        print(obj_pow_40_list)
        print("以下是最大物镜放大倍数为20倍的病理图像：")
        print(obj_pow_20_list)
        print()
        print("以下是最大物镜放大倍数既不为40倍也不为20倍的病理图像：")
        print(obj_pow_other_list)
        print()
        print("程序将在20s后继续运行")
        print()
        time.sleep(20)


def remove_slash_at_end(s):
    # 检查字符串末尾是否有"/"或"\"，如果有，则删去"/"或"\"
    if s.endswith('/') or s.endswith('\\'):
        s = s[:-1]  # 删除最后一个字符
    return s


def main(input_parameters, intermediate_parameters, constant_parameters):
    if not os.path.exists(input_parameters.output_eval_results_dir):
        os.makedirs(input_parameters.output_eval_results_dir)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # 删除中间缓存文件
    # remove_intermediate_files(intermediate_parameters.slide_dir)

    # 快速创造patches
    patch_save_dir = os.path.join(intermediate_parameters.patches_save_dir, 'patches')
    mask_save_dir = os.path.join(intermediate_parameters.patches_save_dir, 'masks')
    stitch_save_dir = os.path.join(intermediate_parameters.patches_save_dir, 'stitches')

    if constant_parameters.process_list:
        process_list = os.path.join(intermediate_parameters.patches_save_dir, constant_parameters.process_list)

    else:
        process_list = None

    directories = {'source': input_parameters.source_slides_dir,
                   'save_dir': intermediate_parameters.patches_save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if constant_parameters.preset:
        preset_df = pd.read_csv(os.path.join('presets', constant_parameters.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                           patch_size=constant_parameters.patch_size,
                                           step_size=constant_parameters.step_size,
                                           seg=constant_parameters.seg, use_default_params=False, save_mask=True,
                                           stitch=constant_parameters.stitch,
                                           patch_level=constant_parameters.patch_level, patch=constant_parameters.patch,
                                           process_list=process_list, auto_skip=constant_parameters.no_auto_skip)

    # 创造slides_name_csv
    create_slides_name_csv(intermediate_parameters.path, intermediate_parameters.name)

    # 快速提取特征
    print('initializing dataset')
    csv_path = os.path.join(current_dir, intermediate_parameters.csv_path)
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(intermediate_parameters.features_save_dir, exist_ok=True)
    os.makedirs(os.path.join(intermediate_parameters.features_save_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(intermediate_parameters.features_save_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(intermediate_parameters.features_save_dir, 'pt_files'))

    print('loading model checkpoint')
    model = resnet50_baseline(pretrained=True)
    model = model.to(device)

    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(constant_parameters.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(intermediate_parameters.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(input_parameters.source_slides_dir, slide_id + constant_parameters.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not constant_parameters.no_auto_skip2 and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(intermediate_parameters.features_save_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            model=model, batch_size=input_parameters.batch_size, verbose=1,
                                            print_every=20,
                                            custom_downsample=constant_parameters.custom_downsample,
                                            target_patch_size=constant_parameters.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(intermediate_parameters.features_save_dir, 'pt_files', bag_base + '.pt'))
        file.close()

    # 创造dataset csv
    df = generate_dataset_csv(intermediate_parameters.input_dir_path)
    dataset_csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset_csv",
                                    os.path.basename(os.path.normpath(intermediate_parameters.input_dir_path)) + ".csv")
    print(dataset_csv_path)
    csv_dir = os.path.dirname(dataset_csv_path)
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(dataset_csv_path, index=False)

    # 在results/Model中创造splits_*.csv
    input_file = intermediate_parameters.input1
    output_dir = intermediate_parameters.output1
    num_files = intermediate_parameters.num_files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    create_csv_files(os.path.join(current_dir, input_file), os.path.join(current_dir, output_dir), num_files)

    # Predict eval 预测
    intermediate_parameters.save_dir = os.path.join(str(input_parameters.output_eval_results_dir), 'eval_results',
                                                    'EVAL_' + str(intermediate_parameters.save_exp_code))
    intermediate_parameters.models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                      constant_parameters.results_dir,
                                                      str(intermediate_parameters.models_exp_code))

    os.makedirs(intermediate_parameters.save_dir, exist_ok=True)

    if constant_parameters.splits_dir is None:
        constant_parameters.splits_dir = intermediate_parameters.models_dir

    assert os.path.isdir(intermediate_parameters.models_dir)
    assert os.path.isdir(constant_parameters.splits_dir)

    settings = {'task': intermediate_parameters.task,
                'split': constant_parameters.split,
                'save_dir': intermediate_parameters.save_dir,
                'models_dir': intermediate_parameters.models_dir,
                'model_type': intermediate_parameters.model_type,
                'drop_out': intermediate_parameters.drop_out,
                'dropout_prop': intermediate_parameters.dropout_prop,
                'model_size': intermediate_parameters.model_size}

    with open(
            intermediate_parameters.save_dir + '/eval_experiment_{}.txt'.format(intermediate_parameters.save_exp_code),
            'w') as f:
        print(settings, file=f)
    f.close()

    print(settings)

    dataset_csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset_csv",
                                    intermediate_parameters.save_exp_code + ".csv")
    # if intermediate_parameters.task == 'Model_example':
    #     intermediate_parameters.n_classes = 2
    #     dataset = Generic_MIL_Dataset(csv_path=dataset_csv_path,
    #                                   data_dir=os.path.join(intermediate_parameters.data_root_dir, ''),
    #                                   shuffle=False, print_info=True, label_dict={'negative': 0, 'positive': 1},
    #                                   patient_strat=False, ignore=[])

    if intermediate_parameters.task == "Model_FGFR_BLCA_Origimed_TCGA_for_biggest_40x_objective":
        intermediate_parameters.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path=dataset_csv_path,
                                      data_dir=os.path.join(intermediate_parameters.data_root_dir, ''),
                                      shuffle=False, print_info=True, label_dict={'negative': 0, "positive": 1},
                                      patient_strat=False, ignore=[])

    elif intermediate_parameters.task == "Model_FGFR_BLCA_Origimed_TCGA_for_biggest_20x_objective":
        intermediate_parameters.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path=dataset_csv_path,
                                      data_dir=os.path.join(intermediate_parameters.data_root_dir, ''),
                                      shuffle=False, print_info=True, label_dict={'negative': 0, "positive": 1},
                                      patient_strat=False, ignore=[])

    elif intermediate_parameters.task == "Model_MET_NSCLC_for_biggest_40x_objective":
        intermediate_parameters.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path=dataset_csv_path,
                                      data_dir=os.path.join(intermediate_parameters.data_root_dir, ''),
                                      shuffle=False, print_info=True, label_dict={'positive': 0, "negative": 1},
                                      patient_strat=False, ignore=[])
        
    elif intermediate_parameters.task == "Model_MET_NSCLC_for_biggest_20x_objective":
        intermediate_parameters.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path=dataset_csv_path,
                                      data_dir=os.path.join(intermediate_parameters.data_root_dir, ''),
                                      shuffle=False, print_info=True, label_dict={'positive': 0, "negative": 1},
                                      patient_strat=False, ignore=[])
    else:
        raise NotImplementedError

    if constant_parameters.k_start == -1:
        start = 0
    else:
        start = constant_parameters.k_start
    if constant_parameters.k_end == -1:
        end = intermediate_parameters.k
    else:
        end = constant_parameters.k_end
    if constant_parameters.fold == -1:
        folds = range(start, end)
    else:
        folds = range(constant_parameters.fold, constant_parameters.fold + 1)
    ckpt_paths = [os.path.join(intermediate_parameters.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
    datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

    start_time = time.time()
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[constant_parameters.split] < 0:
            split_dataset = dataset
        else:
            splits_csv_path = "{}/splits_{}.csv".format(constant_parameters.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=splits_csv_path)
            split_dataset = datasets[datasets_id[constant_parameters.split]]

        # 转换argparse.Namespace到字典
        input_parameters_dict = vars(input_parameters)
        intermediate_parameters_dict = vars(intermediate_parameters)
        constant_parameters_dict = vars(constant_parameters)
        merged_args_dict = {**input_parameters_dict, **intermediate_parameters_dict, **constant_parameters_dict}
        args = argparse.Namespace(**merged_args_dict)
        model, patient_results, test_error, auc, df = eval(split_dataset, args, ckpt_paths[ckpt_idx])

        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1 - test_error)
        df.to_csv(os.path.join(intermediate_parameters.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != intermediate_parameters.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(intermediate_parameters.save_dir, save_name))

    # Clear the screen
    if os.name == "nt":
        subprocess.run(
            f"cls", shell=True, check=True
        )
    elif os.name == "posix":
        subprocess.run(
            f"clear", shell=True, check=True
        )
    else:
        pass

    # 检查切片的最大物镜放大倍数
    check_maximum_resolution(input_parameters)

    # 检查切片特征对应关系
    check_slides_features_correspondence(input_parameters, intermediate_parameters)

    # Ensemble prediction results
    kfiles = return_k_files(intermediate_parameters.input_eval_dir_path)
    result = average_predictions(kfiles, input_parameters.threshold)
    print("以下是预测结果：")
    print(result)
    result.to_csv(os.path.join(intermediate_parameters.input_eval_dir_path, constant_parameters.output2), index=False)
    for i in os.listdir(intermediate_parameters.input_eval_dir_path):
        if i[:15] == "eval_experiment":
            os.remove(os.path.join(intermediate_parameters.input_eval_dir_path, i))
        elif i == "summary.csv":
            os.remove(os.path.join(intermediate_parameters.input_eval_dir_path, i))
        else:
            pass

    abspath = os.path.join(intermediate_parameters.input_eval_dir_path, constant_parameters.output2)
    print(f"整合后的预测结果已保存到文件：{abspath}")
    print()

    # 移除中间生成的文件
    remove_intermediate_files(intermediate_parameters.slide_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于至本数据库和TCGA数据库的尿路上皮癌FGFR致癌突变预测模型，前两个参数必填")

    # Create_patches_fp 中的输入参数
    parser.add_argument("--source_slides_dir", type=str, required=True, help="请输入svs格式病理图像*所在目录*的*完整*路径，例如："
                                                                             "'D:/where/you/store/your/slides', 目录中"
                                                                             "不应有其他格式的文件，目录中不应带中文字符，否则可能会出错")

    # Eval_ensemble 中的输入参数
    parser.add_argument("--output_eval_results_dir", type=str, required=True, help="请告知结果输出文件的*存放目录*的完整路径，例如："
                                                                                   "'D:/where/you/store/your/results'")
    parser.add_argument("--threshold", type=float, default=0.5, help="请根据业务需求确定区分样本是阳性还是阴性的cutoff阈值，默认为0.5，但是该值有很大可能不是你所需要的阈值")
    
    # Generate_eval_splits 中的输入参数
    parser.add_argument("--model_name", type=str, default="Model_FGFR_BLCA_Origimed_TCGA_for_biggest_40x_objective", 
                        help="""请输入所使用的模型名称，默认模型是提供给最大放大倍数为40倍的膀胱癌病理图像，如果你提供的是最大放大倍数为20倍的膀胱癌病理图像，请填写'Model_FGFR_BLCA_Origimed_TCGA_for_biggest_20x_objective'，这是用来预测FGFR变异的；如果要用来预测肺腺癌的MET变异情况，请使用Model_MET_NSCLC_for_biggest_40x_objective或Model_MET_NSCLC_for_biggest_20x_objective模型""")

    # 其他输入参数
    # parser.add_argument("--gpu_num", type=str, default=0, help="请指定要使用的GPU，默认为0，可以指定双GPU以加速程序运行，默认为0,1")
    # parser.add_argument("--batch_size", type=int, default=128, help="请确定批量大小，默认值为128，"
    #                                                                 "如果单GPU显存为6GB，建议值为128；"
    #                                                                 "如果单GPU显存为10GB，建议值为256；"
    #                                                                 "如果单GPU显存为16GB，建议值为512；"
    #                                                                 "单GPU显存如果为其他值，请自行斟酌，"
    #                                                                 "可通过在任务管理器中观察显存用量逐渐调高此值，"
    #                                                                 "在显卡可承受的范围内，batch_size越大，程序运行速度越快，"
    #                                                                 "但如果超出可承受范围，可能会出现图形界面卡顿的情况")

    input_args = parser.parse_args()
    

    # 判定输入参数是否合规
    # 判定args.source_slides_dir是否合规
    if not os.path.isdir(input_args.source_slides_dir):
        raise Exception("输入的source_slides_dir并非是目录路径，请重新确定")

    input_args.source_slides_dir = remove_slash_at_end(input_args.source_slides_dir)
    input_args.batch_size = 128

    """中间参数"""
    # Create_patches_fp 中的中间参数
    project_name = os.path.basename(input_args.source_slides_dir)
    patches_save_dir = os.path.join("./PATCHES", project_name)

    # Create_slides_name_csv 中的中间参数
    path = os.path.join(patches_save_dir, "patches")
    name = project_name

    # Extract_features_fp 中的中间参数
    data_h5_dir = patches_save_dir
    csv_path = os.path.join("slides_name_csv", project_name + "_slides_name.csv")
    features_save_dir = os.path.join("./FEATURES", project_name)

    # Generate_dataset_csv 中的中间参数
    input_dir_path = input_args.source_slides_dir

    # Generate_eval_splits 中的中间参数
    input1 = os.path.join("./dataset_csv", project_name + ".csv")
    output1 = os.path.join("./results", input_args.model_name)
    # if input_args.model_name == "Model_example":
    #     num_files = 10
    if input_args.model_name == 'Model_FGFR_BLCA_Origimed_TCGA_for_biggest_40x_objective':
        num_files = 5
    elif input_args.model_name == "Model_FGFR_BLCA_Origimed_TCGA_for_biggest_20x_objective":
        num_files = 5
    elif input_args.model_name == "Model_MET_NSCLC_for_biggest_40x_objective":
        num_files = 5
    elif input_args.model_name == "Model_MET_NSCLC_for_biggest_20x_objective":
        num_files = 5
    else:
        raise Exception("该模型并不存在！")

    # Eval 中的中间参数
    data_root_dir = features_save_dir
    save_exp_code = project_name
    models_exp_code = input_args.model_name
    # if models_exp_code == "Model_example":
    #     drop_out = True
    #     patch_level = 0
    #     custom_downsample = 1
    #     dropout_prop = 0.65
    #     task = "Model_example"
    #     model_size = "small"
    #     model_type = "clam_sb"
    #     k = 10
    if models_exp_code == "Model_FGFR_BLCA_Origimed_TCGA_for_biggest_40x_objective":
        drop_out = True
        patch_level = 0
        custom_downsample = 2
        dropout_prop = 0.5
        task = "Model_FGFR_BLCA_Origimed_TCGA_for_biggest_40x_objective"
        model_size = "small"
        model_type = "clam_sb"
        k = 5
    elif models_exp_code == "Model_FGFR_BLCA_Origimed_TCGA_for_biggest_20x_objective":
        drop_out = True
        patch_level = 0
        custom_downsample = 1
        dropout_prop = 0.5
        task = "Model_FGFR_BLCA_Origimed_TCGA_for_biggest_20x_objective"
        model_size = "small"
        model_type = "clam_sb"
        k = 5
    elif models_exp_code == "Model_MET_NSCLC_for_biggest_40x_objective":
        drop_out = True
        patch_level = 0
        custom_downsample = 2
        dropout_prop = 0.5
        task = "Model_MET_NSCLC_for_biggest_40x_objective"
        model_size = "small"
        model_type = "clam_sb"
        k=5
    elif models_exp_code == "Model_MET_NSCLC_for_biggest_20x_objective":
        drop_out = True
        patch_level = 0
        custom_downsample = 1
        dropout_prop = 0.5
        task = "Model_MET_NSCLC_for_biggest_20x_objective"
        model_size = "small"
        model_type = "clam_sb"
        k=5

    else:
        raise Exception("该模型并不存在")

    # Eval_ensemble 中的中间参数
    input_eval_dir_path = os.path.join(input_args.output_eval_results_dir, "eval_results", "EVAL_" + project_name)

    # Delete_intermediate_files 中的中间参数
    slide_dir = input_args.source_slides_dir

    """固定参数"""
    # Create_patches_fp 中的固定参数
    patch_size = 256
    preset = "bwh_shigb_fgfr.csv"
    step_size = 256
    process_list = None
    patch = True
    seg = True
    stitch = True
    no_auto_skip = True

    # Extract_features_fp 中的固定参数
    slide_ext = ".svs"
    # custom_downsample = 1
    target_patch_size = -1
    no_auto_skip2 = True

    # Eval 中的固定参数
    results_dir = "./results"
    splits_dir = None
    split = "test"
    k_start = -1
    k_end = -1
    fold = -1

    # Eval_ensemble 中的固定参数
    output2 = "prediction_results.csv"

    intermediate_args = Params(patches_save_dir=patches_save_dir,
                               path=path, name=name,
                               data_h5_dir=data_h5_dir, csv_path=csv_path, features_save_dir=features_save_dir,
                               input_dir_path=input_dir_path,
                               input1=input1, output1=output1, num_files=num_files,
                               data_root_dir=data_root_dir, save_exp_code=save_exp_code,
                               models_exp_code=models_exp_code, drop_out=drop_out,
                               dropout_prop=dropout_prop, task=task, model_size=model_size, model_type=model_type, k=k,
                               input_eval_dir_path=input_eval_dir_path,
                               slide_dir=slide_dir)
    constant_args = Params(step_size=step_size, patch_size=patch_size, patch=patch, seg=seg, stitch=stitch,
                           no_auto_skip=no_auto_skip, preset=preset, patch_level=patch_level, process_list=process_list,
                           slide_ext=slide_ext, custom_downsample=custom_downsample,
                           target_patch_size=target_patch_size, no_auto_skip2=no_auto_skip2,
                           results_dir=results_dir, splits_dir=splits_dir, split=split, k_start=k_start, k_end=k_end,
                           fold=fold,
                           output2=output2)

    check_source_slides_dir(input_args)

    main(input_args, intermediate_args, constant_args)
