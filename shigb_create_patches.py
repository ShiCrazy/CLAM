import argparse
import os
from wsi_preprocessing import tupac



def create_patches(args):
    """
    提取目录中所有slides图像的特征
    args:
        BASE_DIR, SRC_TRAIN_DIR, TRAIN_PREFIX, ROW_TILE_SIZE,
        COL_TILE_SIZE, PATCH_LEVEL, CUSTOM_DOWNSAMPLE
    """
    tupac.BASE_DIR = args.BASE_DIR
    tupac.update_constants()
    tupac.SRC_TRAIN_DIR = args.SRC_TRAIN_DIR
    tupac.TRAIN_PREFIX = args.TRAIN_PREFIX + "-"
    tupac.ROW_TILE_SIZE = args.ROW_COL_TILE_SIZE
    tupac.COL_TILE_SIZE = args.ROW_COL_TILE_SIZE
    tupac.PATCH_LEVEL = args.PATCH_LEVEL
    tupac.CUSTOM_DOWNSAMPLE = args.CUSTOM_DOWNSAMPLE

    if not os.path.exists(tupac.BASE_DIR):
        os.makedirs(tupac.BASE_DIR)
    if args.slide_info:
        tupac.slide_info()
    if args.stage in [0,1,2,3]:
        pass
    if args.stage in [1,2,3]:
        # 将slides转换成images
        tupac.multiprocess_training_slides_to_images()
    if args.stage in [2,3]:
        # 将images添加mask
        tupac.multiprocess_apply_filters_to_images()
    if args.stage == 3:
        # 从images中提取patches
        tupac.multiprocess_filtered_images_to_tiles()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creat Patches in TUPAC way")
    parser.add_argument("--BASE_DIR", type=str, help="Image, filter, tiles结果输出的目录")
    parser.add_argument("--SRC_TRAIN_DIR", type=str, help="Slides结果所在的目录")
    parser.add_argument("--TRAIN_PREFIX", type=str, default="", help="结果文件的前缀")
    parser.add_argument("--ROW_COL_TILE_SIZE", type=int, 
                        help="此为原始tile的大小, 真实tile的大小为ROW_COL_TILE_SIZE/CUSTOM_DOWNSAMPLE")
    parser.add_argument("--SCALE_FACTOR", type=int, default=64, help="指定的下采样倍数")
    parser.add_argument("--PATCH_LEVEL", type=int, default=0, help="切割patch的分辨率层级")
    parser.add_argument("--CUSTOM_DOWNSAMPLE", type=int, default=1, help="自定义下采样的层级")
    parser.add_argument("--stage", type=int, choices=[0,1,2,3], default=3, help="请选择要进行的操作, 0不进行任何操作, 1仅进行简图生成, 2进行简图生成和滤镜添加, 3进行patch生成")
    parser.add_argument("--slide_info", action="store_true", help="是否显示最大放大倍数信息")
    args = parser.parse_args()
    create_patches(args)



