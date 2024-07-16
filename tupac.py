import datetime
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, glob, math, PIL
import openslide
import re
import multiprocessing
import skimage.morphology as sk_morphology
from enum import Enum
import matplotlib.pyplot as plt
import skimage.color as sk_color
from torchvision.transforms import ToTensor

BASE_DIR = "?"
SRC_TRAIN_DIR = "?"
ROW_TILE_SIZE = 512
COL_TILE_SIZE = 512
PATCH_LEVEL = 0
CUSTOM_DOWNSAMPLE = 1

SRC_TRAIN_EXT = "svs"
DEST_TRAIN_EXT = "png"
DEST_TRAIN_DIR = os.path.join(BASE_DIR, "training_" + DEST_TRAIN_EXT)
SCALE_FACTOR = 64
TRAIN_PREFIX = "?"
DEST_TRAIN_SUFFIX = ""
FILTER_DIR = os.path.join(BASE_DIR, "filter_" + DEST_TRAIN_EXT)
THUMBNAIL_EXT = 'jpg'
FILTER_SUFFIX = ""
FILTER_RESULT_TEXT = "filtered"


TISSUE_HIGH_THRESH =80
TISSUE_LOW_THRESH =10
NUM_TOP_TILES = 50

SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
DISPLAY_TILE_SUMMARY_LABELS = False
FONT_PATH = "./Arial Bold.ttf"
SUMMARY_TITLE_FONT_PATH = "./Times-New-Romance-Bold.ttf"

TILE_LABEL_TEXT_SIZE = 10
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
LABEL_ALL_TILES_IN_GOOD_TILE_SUMMARY = False
LABEL_GOOD_TILES_IN_GOOD_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_GOOD_TILE_SUMMARY = False


TILE_BORDER_SIZE = 1
HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

TILE_SUMMARY_DIR = os.path.join(BASE_DIR, "tile_summary_" + DEST_TRAIN_EXT)
TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_" + DEST_TRAIN_EXT)

GOOD_TILES_SUFFIX = "good_tile_summary"
GOOD_TILES_DIR = os.path.join(BASE_DIR, GOOD_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
GOOD_TILES_THUMBNAIL_DIR = os.path.join(BASE_DIR, GOOD_TILES_SUFFIX + "_thumbnail_" + THUMBNAIL_EXT)
GOOD_TILES_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, GOOD_TILES_SUFFIX + "_on_original_" + DEST_TRAIN_EXT)
GOOD_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR,
                                                    GOOD_TILES_SUFFIX + "_on_original_thumbnail_" + THUMBNAIL_EXT)

HSV_PURPLE = 270
HSV_PINK = 330

TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

TILE_TEXT_SIZE = 36
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)

TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data")
TILE_DATA_SUFFIX = "tile_data"
TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TRAIN_EXT)
TILE_SUFFIX = "tile"


def update_constants():
    global DEST_TRAIN_DIR, FILTER_DIR, TILE_SUMMARY_DIR, TILE_SUMMARY_ON_ORIGINAL_DIR, GOOD_TILES_DIR, GOOD_TILES_THUMBNAIL_DIR, GOOD_TILES_ON_ORIGINAL_DIR, GOOD_TILES_ON_ORIGINAL_THUMBNAIL_DIR, TILE_DATA_DIR, TILE_DIR
    DEST_TRAIN_DIR = os.path.join(BASE_DIR, "training_" + DEST_TRAIN_EXT)
    FILTER_DIR = os.path.join(BASE_DIR, "filter_" + DEST_TRAIN_EXT)
    TILE_SUMMARY_DIR = os.path.join(BASE_DIR, "tile_summary_" + DEST_TRAIN_EXT)
    TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_" + DEST_TRAIN_EXT)
    GOOD_TILES_DIR = os.path.join(BASE_DIR, GOOD_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
    GOOD_TILES_THUMBNAIL_DIR = os.path.join(BASE_DIR, GOOD_TILES_SUFFIX + "_thumbnail_" + THUMBNAIL_EXT)
    GOOD_TILES_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, GOOD_TILES_SUFFIX + "_on_original_" + DEST_TRAIN_EXT)
    GOOD_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR,
                                                        GOOD_TILES_SUFFIX + "_on_original_thumbnail_" + THUMBNAIL_EXT)
    TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data")
    TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TRAIN_EXT)


class Time:

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        end = datetime.datetime.now()
        time_elapsed = end - self.start
        return time_elapsed


def summary_title(tile_summary):
    return "Slide %03d Tile Summary:" % tile_summary.slide_num

def summary_stats(tile_summary):
    return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
           "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
           "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
           "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
           "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
           "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
               tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
           "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
           " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
               tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
           " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
               tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
               TISSUE_HIGH_THRESH) + \
           " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
               tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
           " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)

class TileSummary:
    """
  Class for tile summary information.
  """

    slide_num = None
    orig_w = None
    orig_h = None
    orig_tile_w = None
    orig_tile_h = None
    scale_factor = SCALE_FACTOR
    scaled_w = None
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    # mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0

    def __init__(self, slide_num, slide_filepath, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h,
                 scaled_tile_w,
                 scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
        self.slide_num = slide_num
        self.slide_filepath = slide_filepath
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        self.tissue_percentage = tissue_percentage
        self.num_col_tiles = num_col_tiles
        self.num_row_tiles = num_row_tiles
        self.tiles = []

    def __str__(self):
        return summary_title(self) + "\n" + summary_stats(self)

    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def num_tiles(self):
        return self.num_row_tiles * self.num_col_tiles

    def tiles_by_tissue_percentage(self):
        sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
        return sorted_list

    def tiles_by_score(self):
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list

    def top_tiles(self):
        sorted_tiles = self.tiles_by_score()
        top_tiles = sorted_tiles[:NUM_TOP_TILES]
        return top_tiles

    def good_tiles(self):
        sorted_tiles = self.tiles_by_score()
        sorted_score_list = []
        No_of_first_tile_scoring_less_threshold = 0
        for i in sorted_tiles:
            a = float(str(i)[-7:-1])
            sorted_score_list.append(a)

        for index, x in enumerate(sorted_score_list):
            if x < 0.7:
                No_of_first_tile_scoring_less_threshold = index
                break
        # print(No_of_first_tile_scoring_less_threshold)
        good_tiles = sorted_tiles[:No_of_first_tile_scoring_less_threshold]
        print(good_tiles)
        return good_tiles

    def get_tile(self, row, col):
        tile_index = (row - 1) * self.num_col_tiles + (col - 1)
        tile = self.tiles[tile_index]
        return tile

    def display_summaries(self):
        summary_and_tiles(self.slide_num, self.slide_filepath, display=True, save_summary=False, save_data=False, save_top_tiles=False)


def get_num_training_slides():

    num_training_slides = len(glob.glob1(SRC_TRAIN_DIR, "*." + SRC_TRAIN_EXT))
    return num_training_slides

def training_slide_range_to_images(start_ind, end_ind):

    for slide_num in range(start_ind, end_ind + 1):
        slide_filepath = os.path.join(SRC_TRAIN_DIR, os.listdir(SRC_TRAIN_DIR)[slide_num - 1])
        if slide_filepath[-4:] != ".svs":
            continue
        training_slide_to_image(slide_num, slide_filepath)
    return start_ind, end_ind

def training_slide_to_image(slide_number, slide_filepath):

    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number, slide_filepath)

    img_path = get_training_image_path(slide_number, slide_filepath, large_w, large_h, new_w, new_h)
    print("Saving image to: " + img_path)
    if not os.path.exists(DEST_TRAIN_DIR):
        os.makedirs(DEST_TRAIN_DIR)
    img.save(img_path)

def slide_to_scaled_pil_image(slide_number, slide_filepath):

    # slide_filepath = get_training_slide_path(slide_number)
    print("Opening Slide #%d: %s" % (slide_number, slide_filepath))
    slide = open_slide(slide_filepath)

    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    # levels = slide.level_count
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    print("level", level)
    print("slide.level_dimensions[level]",slide.level_dimensions[level])
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, large_w, large_h, new_w, new_h

def get_training_image_path(slide_number, slide_filepath, large_w=None, large_h=None, small_w=None, small_h=None):
    padded_sl_num = str(slide_number).zfill(3)
    slide_filepath = slide_filepath.replace("/", "\\")
    slide_name = slide_filepath.split("\\")[-1][:-4]

    if large_w is None and large_h is None and small_w is None and small_h is None:
        wildcard_path = os.path.join(DEST_TRAIN_DIR,
                                     TRAIN_PREFIX + padded_sl_num + "-" + slide_name + "*." + DEST_TRAIN_EXT)
        img_path = glob.glob(wildcard_path)[0]
    else:
        img_path = os.path.join(DEST_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "-" + slide_name + "-" + str(
            SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
            large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + DEST_TRAIN_EXT)
    return img_path

def open_slide(filename):
    try:
        slide = openslide.open_slide(filename)
    except openslide.OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide


def apply_filters_to_image_list(image_num_list, save, display):

    html_page_info = dict()
    for slide_num in image_num_list:
        slide_filepath = os.path.join(SRC_TRAIN_DIR, os.listdir(SRC_TRAIN_DIR)[slide_num - 1])
        if slide_filepath[-4:] != ".svs":
            continue
        _, info = apply_filters_to_image(slide_num, slide_filepath, save=save, display=display)
        html_page_info.update(info)
    return image_num_list, html_page_info

def apply_filters_to_image_range(start_ind, end_ind, save, display):
    html_page_info = dict()
    for slide_num in range(start_ind, end_ind + 1):
        slide_filepath = os.path.join(SRC_TRAIN_DIR, os.listdir(SRC_TRAIN_DIR)[slide_num - 1])
        print(slide_filepath)
        if slide_filepath[-4:] != ".svs":
            continue
        _, info = apply_filters_to_image(slide_num, slide_filepath, save, display=display)
        html_page_info.update(info)
    return start_ind, end_ind, html_page_info

def apply_filters_to_image(slide_num, slide_filepath, save=True, display=False):
    t = Time()
    print("Processing slide #%d" % slide_num)

    info = dict()

    if save and not os.path.exists(FILTER_DIR):
        os.makedirs(FILTER_DIR)
    img_path = get_training_image_path(slide_num, slide_filepath)
    np_orig = open_image_np(img_path)
    filtered_np_img = apply_image_filters(np_orig, slide_filepath, slide_num, info, save=save, display=display)

    if save:
        t1 = Time()
        result_path = get_filter_image_result(slide_num, slide_filepath)
        pil_img = np_to_pil(filtered_np_img)
        pil_img.save(result_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

    print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

    return filtered_np_img, info

def open_image_np(filename):
    pil_img = open_image(filename)
    np_img = pil_to_np_rgb(pil_img)
    return np_img

def open_image(filename):
    image = Image.open(filename)
    return image



def apply_image_filters(np_img, slide_filepath, slide_num=None, info=None, save=False, display=False):

    rgb = np_img
    save_display(save, display, info, rgb, slide_filepath, slide_num, 1, "Original", "rgb")

    mask_not_green = filter_green_channel(rgb)

    mask_not_gray = filter_grays(rgb)

    mask_no_red_pen = filter_red_pen(rgb)

    mask_no_green_pen = filter_green_pen(rgb)

    mask_no_blue_pen = filter_blue_pen(rgb)

    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen

    mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
    rgb_remove_small = mask_rgb(rgb, mask_remove_small)
    # save_display(save, display, info, rgb_remove_small, slide_num, 8,
    #              "Not Gray, Not Green, No Pens,\nRemove Small Objects",
    #              "rgb-not-green-not-gray-no-pens-remove-small")

    img = rgb_remove_small
    return img

def save_display(save, display, info, np_img, slide_filepath, slide_num, filter_num, display_text, file_text,
                 display_mask_percentage=True):
    mask_percentage = None
    if display_mask_percentage:
        mask_percentage = mask_percent(np_img)
        display_text = display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
    if slide_num is None and filter_num is None:
        pass
    elif filter_num is None:
        display_text = "S%03d " % slide_num + display_text
    elif slide_num is None:
        display_text = "F%03d " % filter_num + display_text
    else:
        display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
    if display:
        display_img(np_img, display_text)
    if save:
        save_filtered_image(np_img, slide_filepath, slide_num, filter_num, file_text)
    if info is not None:
        info[slide_num * 1000 + filter_num] = (slide_num, filter_num, display_text, file_text, mask_percentage)

def get_filter_image_result(slide_number, slide_filepath):
    padded_sl_num = str(slide_number).zfill(3)
    slide_filepath = slide_filepath.replace("/", "\\")
    slide_name = slide_filepath.split("\\")[-1][:-4]
    training_img_path = get_training_image_path(slide_number, slide_filepath)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    img_path = os.path.join(FILTER_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        SCALE_FACTOR) + "x-" + FILTER_SUFFIX + "-" + slide_name + "-" + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
        small_h) + "-" + FILTER_RESULT_TEXT + "." + DEST_TRAIN_EXT)
    return img_path

def np_to_pil(np_img):
  
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)

def pil_to_np_rgb(pil_img):
  
    t = Time()
    rgb = np.asarray(pil_img)
    np_info(rgb, "RGB", t.elapsed())
    return rgb

def np_info(np_arr, name=None, elapsed=None):

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    ADDITIONAL_NP_STATS = False
    if ADDITIONAL_NP_STATS is False:
        print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max_value = np_arr.max()
        min_value = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
        name, str(elapsed), min_value, max_value, mean, is_binary, np_arr.dtype, np_arr.shape))

def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):

    t = Time()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
                mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Filter Green Channel", t.elapsed())
    return np_img

def filter_grays(rgb, tolerance=15, output_type="bool"):



    t = Time()
    # (h, w, c) = rgb.shape

    rgb = rgb.astype(int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Grays", t.elapsed())
    return result

def filter_red_pen(rgb, output_type="bool"):
    t = Time()
    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
             filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
             filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
             filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
             filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
             filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
             filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
             filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
             filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Red Pen", t.elapsed())
    return result

def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Red", t.elapsed())
    return result

def filter_green_pen(rgb, output_type="bool"):
    t = Time()
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
             filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
             filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
             filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
             filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
             filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
             filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
             filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
             filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
             filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
             filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Green Pen", t.elapsed())
    return result

def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):

    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Green", t.elapsed())
    return result

def filter_blue_pen(rgb, output_type="bool"):
    t = Time()
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
             filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
             filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
             filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
             filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
             filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
             filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
             filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
             filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
             filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Blue Pen", t.elapsed())
    return result

def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Blue", t.elapsed())
    return result

def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    t = Time()

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size / 2
        print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
            mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Remove Small Objs", t.elapsed())
    return np_img

def mask_percent(np_img):
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage

def mask_rgb(rgb, mask):
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    np_info(result, "Mask RGB", t.elapsed())
    return result

def mask_percentage_text(mask_percentage):
    return "%3.2f%%" % mask_percentage

def display_img(np_img, text=None, font_path="", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    if text is not None:
        if font_path == "":
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, size)
        if bg:
            (x, y) = draw.textsize(text, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color, font=font)
    result.show()

def save_filtered_image(np_img, slide_filepath, slide_num, filter_num, filter_text):
    t = Time()
    slide_filepath = slide_filepath.replace("/", "\\")
    slide_name = slide_filepath.split("\\")[-1][:-4]
    filepath = get_filter_image_path(slide_num, slide_name, filter_num, filter_text)
    pil_img = np_to_pil(np_img)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))

def get_filter_image_path(slide_number, slide_name, filter_number, filter_name_info):
    dir_name = FILTER_DIR
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    img_path = os.path.join(dir_name, get_filter_image_filename(slide_number, slide_name, filter_number, filter_name_info))
    return img_path

def get_filter_image_filename(slide_number, slide_name, filter_number, filter_name_info, thumbnail=False):
    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT
    padded_sl_num = str(slide_number).zfill(3)
    padded_fi_num = str(filter_number).zfill(3)
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + padded_fi_num + "-" + FILTER_SUFFIX + "-" + slide_name + "-" + filter_name_info + "." + ext
    return img_filename

def image_list_to_tiles(image_num_list, slide_filepath, display=False, save_summary=False, save_data=False, save_top_tiles=False,
                        save_good_tiles=False):

    tile_summaries_dict = dict()
    for slide_num in image_num_list:
        tile_summary = summary_and_tiles(slide_num, slide_filepath, display, save_summary, save_data, save_top_tiles, save_good_tiles)
        tile_summaries_dict[slide_num] = tile_summary
    return image_num_list, tile_summaries_dict

def image_range_to_tiles(start_ind, end_ind, display=False, save_summary=False, save_data=False, save_top_tiles=False,
                         save_good_tiles=False):
    image_num_list = list()
    tile_summaries_dict = dict()
    for slide_num in range(start_ind, end_ind + 1):
        slide_filepath = os.path.join(SRC_TRAIN_DIR, os.listdir(SRC_TRAIN_DIR)[slide_num - 1])
        if slide_filepath[-4:] != ".svs":
            continue
        print(slide_filepath.split("\\")[-1][:-4])
        tile_summary = summary_and_tiles(slide_num, slide_filepath, display, save_summary, save_data, save_top_tiles,
                                         save_good_tiles)
        image_num_list.append(slide_num)
        tile_summaries_dict[slide_num] = tile_summary
    return image_num_list, tile_summaries_dict

def summary_and_tiles(slide_num, slide_filepath, display=False, save_summary=True, save_data=True,
                      save_top_tiles=False, save_good_tiles=True):
    img_path = get_filter_image_result(slide_num, slide_filepath)
    np_img = open_image_np(img_path)

    tile_sum = score_tiles(slide_num, slide_filepath, np_img)
    if tile_sum:
        print("shigbshigbshigbsibsiisgihsgihsighisghisghisghis")
    else:
        print("ggggggggggggggggggggggggggggggggggggggggggggggg")
    if save_data:
        save_tile_data(tile_sum)
    generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
    generate_good_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
    if save_top_tiles:
        for tile in tile_sum.top_tiles():
            tile.save_tile()
    if save_good_tiles:
        for tile in tile_sum.good_tiles():
            print(tile)
            tile.save_tile()        
    return tile_sum

def tissue_percent(np_img):

    return 100 - mask_percent(np_img)

def score_tiles(slide_num, slide_filepath, np_img=None, dimensions=None, small_tile_in_tile=False):
    if dimensions is None:
        img_path = get_filter_image_result(slide_num, slide_filepath)
        o_w, o_h, w, h = parse_dimensions_from_image_filename(img_path)
    else:
        o_w, o_h, w, h = dimensions

    if np_img is None:
        np_img = open_image_np(img_path)

    row_tile_size = round(ROW_TILE_SIZE / SCALE_FACTOR)  # use round?
    col_tile_size = round(COL_TILE_SIZE / SCALE_FACTOR)  # use round?

    num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

    tile_sum = TileSummary(slide_num=slide_num,
                           slide_filepath=slide_filepath,
                           orig_w=o_w,
                           orig_h=o_h,
                           orig_tile_w=COL_TILE_SIZE,
                           orig_tile_h=ROW_TILE_SIZE,
                           scaled_w=w,
                           scaled_h=h,
                           scaled_tile_w=col_tile_size,
                           scaled_tile_h=row_tile_size,
                           tissue_percentage=tissue_percent(np_img),
                           num_col_tiles=num_col_tiles,
                           num_row_tiles=num_row_tiles)

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0
    tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
    for t in tile_indices:
        count += 1  # tile_num
        r_s, r_e, c_s, c_e, r, c = t
        np_tile = np_img[r_s:r_e, c_s:c_e]
        t_p = tissue_percent(np_tile)
        amount = tissue_quantity(t_p)
        if amount == TissueQuantity.HIGH:
            high += 1
        elif amount == TissueQuantity.MEDIUM:
            medium += 1
        elif amount == TissueQuantity.LOW:
            low += 1
        elif amount == TissueQuantity.NONE:
            none += 1
        o_c_s, o_r_s = small_to_large_mapping((c_s, r_s), (o_w, o_h))
        o_c_e, o_r_e = small_to_large_mapping((c_e, r_e), (o_w, o_h))

        # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
        if (o_c_e - o_c_s) > COL_TILE_SIZE:
            o_c_e -= 1
        if (o_r_e - o_r_s) > ROW_TILE_SIZE:
            o_r_e -= 1

        score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, slide_num, r, c)

        np_scaled_tile = np_tile if small_tile_in_tile else None
        tile = Tile(tile_sum, slide_num, slide_filepath, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e,
                    o_c_s, o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
        tile_sum.tiles.append(tile)

    tile_sum.count = count
    tile_sum.high = high
    tile_sum.medium = medium
    tile_sum.low = low
    tile_sum.none = none

    tiles_by_score = tile_sum.tiles_by_score()
    rank = 0
    for t in tiles_by_score:
        rank += 1
        t.rank = rank

    return tile_sum

def save_tile_data(tile_summary):

    time = Time()

    csv = summary_title(tile_summary) + "\n" + summary_stats(tile_summary)

    csv += "\n\n\nTile Num,Row,Column,Tissue %,Tissue Quantity,Col Start,Row Start,Col End,Row End,Col Size,Row Size," + \
           "Original Col Start,Original Row Start,Original Col End,Original Row End,Original Col Size,Original Row Size," + \
           "Color Factor,S and V Factor,Quantity Factor,Score\n"

    for t in tile_summary.tiles:
        line = "%d,%d,%d,%4.2f,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%4.0f,%4.2f,%4.2f,%0.4f\n" % (
            t.tile_num, t.r, t.c, t.tissue_percentage, t.tissue_quantity().name, t.c_s, t.r_s, t.c_e, t.r_e,
            t.c_e - t.c_s,
            t.r_e - t.r_s, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s, t.color_factor,
            t.s_and_v_factor, t.quantity_factor, t.score)
        csv += line

    data_path = get_tile_data_path(tile_summary.slide_num, tile_summary.slide_filepath)
    csv_file = open(data_path, "w", encoding="utf-8")
    csv_file.write(csv)
    csv_file.close()

    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Data", str(time.elapsed()), data_path))

def generate_tile_summaries(tile_sum, np_img, display=True, save_summary=False):

    z = 300  # height of area at top of summary slide
    slide_num = tile_sum.slide_num
    slide_filepath = tile_sum.slide_filepath
    rows = tile_sum.scaled_h
    cols = tile_sum.scaled_w
    row_tile_size = tile_sum.scaled_tile_h
    col_tile_size = tile_sum.scaled_tile_w
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw = ImageDraw.Draw(summary)

    original_img_path = get_training_image_path(slide_num, slide_filepath)
    np_orig = open_image_np(original_img_path)
    summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw_orig = ImageDraw.Draw(summary_orig)

    for t in tile_sum.tiles:
        border_color = tile_border_color(t.tissue_percentage)
        tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
        tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

    summary_txt = summary_title(tile_sum) + "\n" + summary_stats(tile_sum)

    # summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)  # 没有这个字体，用下面的默认字体替代
    summary_font = ImageFont.load_default()
    draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
    draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

    if DISPLAY_TILE_SUMMARY_LABELS:
        count = 0
        for t in tile_sum.tiles:
            count += 1
            label = "R%d\nC%d" % (t.r, t.c)
            font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
            # drop shadow behind text
            draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)
            draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)

            draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
            draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

    if display:
        summary.show()
        summary_orig.show()
    if save_summary:
        save_tile_summary_image(summary, slide_num, slide_filepath)
        save_tile_summary_on_original_image(summary_orig, slide_num, slide_filepath)

def generate_good_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_good_stats=False,
                                 label_all_tiles=LABEL_ALL_TILES_IN_GOOD_TILE_SUMMARY,
                                 label_good_tiles=LABEL_GOOD_TILES_IN_GOOD_TILE_SUMMARY,
                                 border_all_tiles=BORDER_ALL_TILES_IN_GOOD_TILE_SUMMARY):
    z = 300  # height of area at good of summary slide
    slide_num = tile_sum.slide_num
    slide_filepath = tile_sum.slide_filepath
    rows = tile_sum.scaled_h
    cols = tile_sum.scaled_w
    row_tile_size = tile_sum.scaled_tile_h
    col_tile_size = tile_sum.scaled_tile_w
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw = ImageDraw.Draw(summary)

    original_img_path = get_training_image_path(slide_num, slide_filepath)
    np_orig = open_image_np(original_img_path)
    summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw_orig = ImageDraw.Draw(summary_orig)

    if border_all_tiles:
        for t in tile_sum.tiles:
            border_color = faded_tile_border_color(t.tissue_percentage)
            tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)
            tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)

    tbs = TILE_BORDER_SIZE
    good_tiles = tile_sum.good_tiles()
    for t in good_tiles:
        border_color = tile_border_color(t.tissue_percentage)
        tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
        tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
        if border_all_tiles:
            tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
            tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

    summary_title_name = "Slide %03d Good Tile Summary:" % slide_num
    summary_txt = summary_title_name + "\n" + summary_stats(tile_sum)

    summary_font = ImageFont.load_default()
    draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
    draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

    tiles_to_label = tile_sum.tiles if label_all_tiles else good_tiles
    h_offset = TILE_BORDER_SIZE + 2
    v_offset = TILE_BORDER_SIZE
    h_ds_offset = TILE_BORDER_SIZE + 3
    v_ds_offset = TILE_BORDER_SIZE + 1

    if label_good_tiles:
        for t in tiles_to_label:
            label = "R%d\nC%d" % (t.r, t.c)
            # font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
            font = ImageFont.load_default()
            # drop shadow behind text
            draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
            draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

            draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
            draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

    if show_good_stats:
        # The below two functions called are right.
        summary = add_tile_stats_to_top_tile_summary(summary, good_tiles, z)
        summary_orig = add_tile_stats_to_top_tile_summary(summary_orig, good_tiles, z)

    if display:
        summary.show()
        summary_orig.show()
    if save_summary:
        save_good_tiles_image(summary, slide_num, slide_filepath)
        save_good_tiles_on_original_image(summary_orig, slide_num, slide_filepath)

def parse_dimensions_from_image_filename(filename):
    m = re.match(r".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
    large_w = int(m.group(1))
    large_h = int(m.group(2))
    small_w = int(m.group(3))
    small_h = int(m.group(4))
    return large_w, large_h, small_w, small_h

def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
    num_row_tiles = math.ceil(rows / row_tile_size)
    num_col_tiles = math.ceil(cols / col_tile_size)
    return num_row_tiles, num_col_tiles

def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
    indices = list()
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    for r in range(0, num_row_tiles):
        start_r = r * row_tile_size
        end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
        for c in range(0, num_col_tiles):
            start_c = c * col_tile_size
            end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
            indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
    return indices

def tissue_quantity(tissue_percentage):
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        return TissueQuantity.HIGH
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        return TissueQuantity.MEDIUM
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        return TissueQuantity.LOW
    else:
        return TissueQuantity.NONE

def tile_to_pil_tile(tile):
    t = tile
    # print(t.slide_filepath)
    s = open_slide(t.slide_filepath)

    x, y = t.o_c_s, t.o_r_s
    w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s

    # 切片的长与宽理应相等，还没有测试，请注意（已测试，这点需要特别注意，有可能不相等，如果不相等就不会保存图像，影响不大，因为后面会调整）
    tile_region = s.read_region((x, y), PATCH_LEVEL, (w, h))
    # RGBA to RGB
    pil_img = tile_region.convert("RGB")
    return pil_img


def tile_to_np_tile(tile):
    pil_img = tile_to_pil_tile(tile)
    np_img = pil_to_np_rgb(pil_img)
    return np_img

def save_display_tile(tile, save=True, display=False):
    tile_pil_img = tile_to_pil_tile(tile)
    if tile_pil_img is not None:
        # if ROW_TILE_SIZE == COL_TILE_SIZE:
        #     tile_pil_img = tile_pil_img.resize((224, 224), Image.ANTIALIAS)
        if save:
            t = Time()
            img_path = get_tile_image_path(tile)
            dir_name = os.path.dirname(img_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            tile_pil_img = tile_pil_img.resize((ROW_TILE_SIZE//CUSTOM_DOWNSAMPLE, COL_TILE_SIZE//CUSTOM_DOWNSAMPLE)) # ！！！！！！！！为了适应HIPT，将切片的大小改为256*256，以实现在20倍放大倍数下patch的效果！！！！！！！！！！！
            tile_pil_img.save(img_path, )
            print("%-20s | Time: %-14s  Name: %s" % ("Save Tile", str(t.elapsed()), img_path))

        if display:
            tile_pil_img.show()

def get_tile_image_path(tile):
    t = tile
    padded_sl_num = str(t.slide_num).zfill(3)
    slide_filepath = t.slide_filepath.replace("/", "\\")
    slide_name = str(slide_filepath).split("\\")[-1][:-4]
    tile_path = os.path.join(TILE_DIR, slide_name,
                             TRAIN_PREFIX + padded_sl_num + "-" + TILE_SUFFIX + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                                 t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s,
                                 t.o_r_e - t.o_r_s) + "." + DEST_TRAIN_EXT)
    return tile_path

def display_tile(tile, rgb_histograms=True, hsv_histograms=True):

    text = "S%03d R%03d C%03d\n" % (tile.slide_num, tile.r, tile.c)
    text += "Score:%4.2f Tissue:%5.2f%% CF:%2.0f SVF:%4.2f QF:%4.2f\n" % (
        tile.score, tile.tissue_percentage, tile.color_factor, tile.s_and_v_factor, tile.quantity_factor)
    text += "Rank #%d of %d" % (tile.rank, tile.tile_summary.num_tiles())

    np_scaled_tile = tile.get_np_scaled_tile()
    if np_scaled_tile is not None:
        small_text = text + "\n \nSmall Tile (%d x %d)" % (np_scaled_tile.shape[1], np_scaled_tile.shape[0])
        if rgb_histograms and hsv_histograms:
            display_image_with_rgb_and_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
        elif rgb_histograms:
            display_image_with_rgb_histograms(np_scaled_tile, small_text, scale_up=True)
        elif hsv_histograms:
            display_image_with_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
        else:
            display_image(np_scaled_tile, small_text, scale_up=True)

    np_tile = tile.get_np_tile()
    text += " based on small tile\n \nLarge Tile (%d x %d)" % (np_tile.shape[1], np_tile.shape[0])
    if rgb_histograms and hsv_histograms:
        display_image_with_rgb_and_hsv_histograms(np_tile, text)
    elif rgb_histograms:
        display_image_with_rgb_histograms(np_tile, text)
    elif hsv_histograms:
        display_image_with_hsv_histograms(np_tile, text)
    else:
        display_image(np_tile, text)

def filter_rgb_to_hsv(np_img, display_np_info=True):
    if display_np_info:
        t = Time()
    hsv = sk_color.rgb2hsv(np_img)
    if display_np_info:
        np_info(hsv, "RGB to HSV", t.elapsed())
    return hsv

def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):

    if display_np_info:
        t = Time()
    h = hsv[:, :, 0]
    h = h.flatten()
    if output_type == "int":
        h *= 360
        h = h.astype("int")
    if display_np_info:
        np_info(hsv, "HSV to H", t.elapsed())
    return h

def filter_hsv_to_s(hsv):
    s = hsv[:, :, 1]
    s = s.flatten()
    return s

def filter_hsv_to_v(hsv):
    v = hsv[:, :, 2]
    v = v.flatten()
    return v


def display_image_with_rgb_and_hsv_histograms(np_rgb, text=None, scale_up=False):
    hsv = filter_rgb_to_hsv(np_rgb)
    np_r = np_rgb_r_histogram(np_rgb)
    np_g = np_rgb_g_histogram(np_rgb)
    np_b = np_rgb_b_histogram(np_rgb)
    np_h = np_hsv_hue_histogram(filter_hsv_to_h(hsv))
    np_s = np_hsv_saturation_histogram(filter_hsv_to_s(hsv))
    np_v = np_hsv_value_histogram(filter_hsv_to_v(hsv))

    r_r, r_c, _ = np_r.shape
    g_r, g_c, _ = np_g.shape
    b_r, b_c, _ = np_b.shape
    h_r, h_c, _ = np_h.shape
    s_r, s_c, _ = np_s.shape
    v_r, v_c, _ = np_v.shape

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i  # for simplicity assign title+image to image
        img_r, img_c, img_ch = np_rgb.shape

    rgb_hists_c = max(r_c, g_c, b_c)
    rgb_hists_r = r_r + g_r + b_r
    rgb_hists = np.zeros([rgb_hists_r, rgb_hists_c, img_ch], dtype=np.uint8)
    rgb_hists[0:r_r, 0:r_c] = np_r
    rgb_hists[r_r:r_r + g_r, 0:g_c] = np_g
    rgb_hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

    hsv_hists_c = max(h_c, s_c, v_c)
    hsv_hists_r = h_r + s_r + v_r
    hsv_hists = np.zeros([hsv_hists_r, hsv_hists_c, img_ch], dtype=np.uint8)
    hsv_hists[0:h_r, 0:h_c] = np_h
    hsv_hists[h_r:h_r + s_r, 0:s_c] = np_s
    hsv_hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

    r = max(img_r, rgb_hists_r, hsv_hists_r)
    c = img_c + rgb_hists_c + hsv_hists_c
    combo = np.zeros([r, c, img_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:img_r, 0:img_c] = np_rgb
    combo[0:rgb_hists_r, img_c:img_c + rgb_hists_c] = rgb_hists
    combo[0:hsv_hists_r, img_c + rgb_hists_c:c] = hsv_hists
    pil_combo = np_to_pil(combo)
    pil_combo.show()



def display_image_with_rgb_histograms(np_rgb, text=None, scale_up=False):
    np_r = np_rgb_r_histogram(np_rgb)
    np_g = np_rgb_g_histogram(np_rgb)
    np_b = np_rgb_b_histogram(np_rgb)
    r_r, r_c, _ = np_r.shape
    g_r, g_c, _ = np_g.shape
    b_r, b_c, _ = np_b.shape

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i  # for simplicity assign title+image to image
        img_r, img_c, img_ch = np_rgb.shape

    hists_c = max(r_c, g_c, b_c)
    hists_r = r_r + g_r + b_r
    hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

    hists[0:r_r, 0:r_c] = np_r
    hists[r_r:r_r + g_r, 0:g_c] = np_g
    hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

    r = max(img_r, hists_r)
    c = img_c + hists_c
    combo = np.zeros([r, c, img_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:img_r, 0:img_c] = np_rgb
    combo[0:hists_r, img_c:c] = hists
    pil_combo = np_to_pil(combo)
    pil_combo.show()

def display_image_with_hsv_histograms(np_rgb, text=None, scale_up=False):
    hsv = filter_rgb_to_hsv(np_rgb)
    np_h = np_hsv_hue_histogram(filter_hsv_to_h(hsv))
    np_s = np_hsv_saturation_histogram(filter_hsv_to_s(hsv))
    np_v = np_hsv_value_histogram(filter_hsv_to_v(hsv))
    h_r, h_c, _ = np_h.shape
    s_r, s_c, _ = np_s.shape
    v_r, v_c, _ = np_v.shape

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i  # for simplicity assign title+image to image
        img_r, img_c, img_ch = np_rgb.shape

    hists_c = max(h_c, s_c, v_c)
    hists_r = h_r + s_r + v_r
    hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

    hists[0:h_r, 0:h_c] = np_h
    hists[h_r:h_r + s_r, 0:s_c] = np_s
    hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

    r = max(img_r, hists_r)
    c = img_c + hists_c
    combo = np.zeros([r, c, img_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:img_r, 0:img_c] = np_rgb
    combo[0:hists_r, img_c:c] = hists
    pil_combo = np_to_pil(combo)
    pil_combo.show()

def display_image(np_rgb, text=None, scale_up=False):

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i

    pil_img = np_to_pil(np_rgb)
    pil_img.show()

def np_rgb_r_histogram(rgb):
    hist = np_rgb_channel_histogram(rgb, 0, "R")
    return hist

def np_rgb_g_histogram(rgb):
    hist = np_rgb_channel_histogram(rgb, 1, "G")
    return hist

def np_rgb_b_histogram(rgb):
    hist = np_rgb_channel_histogram(rgb, 2, "B")
    return hist

def np_hsv_hue_histogram(h):
    figure = plt.figure()
    canvas = figure.canvas
    _, _, patches = plt.hist(h, bins=360)
    plt.title("HSV Hue Histogram, mean=%3.1f, std=%3.1f" % (np.mean(h), np.std(h)))

    bin_num = 0
    for patch in patches:
        rgb_color = colorsys.hsv_to_rgb(bin_num / 360.0, 1, 1)
        patch.set_facecolor(rgb_color)
        bin_num += 1

    canvas.draw()
    w, h = canvas.get_width_height()
    np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(figure)
    np_info(np_hist)
    return np_hist

def np_hsv_saturation_histogram(s):
    title = "HSV Saturation Histogram, mean=%.2f, std=%.2f" % (np.mean(s), np.std(s))
    return np_histogram(s, title)

def np_hsv_value_histogram(v):
    title = "HSV Value Histogram, mean=%.2f, std=%.2f" % (np.mean(v), np.std(v))
    return np_histogram(v, title)



def np_rgb_channel_histogram(rgb, ch_num, ch_name):
    ch = rgb[:, :, ch_num]
    ch = ch.flatten()
    title = "RGB %s Histogram, mean=%.2f, std=%.2f" % (ch_name, np.mean(ch), np.std(ch))
    return np_histogram(ch, title, bins=256)

def np_histogram(data, title, bins="auto"):
    figure = plt.figure()
    canvas = figure.canvas
    plt.hist(data, bins=bins)
    plt.title(title)

    canvas.draw()
    w, h = canvas.get_width_height()
    np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(figure)
    np_info(np_hist)
    return np_hist

class TissueQuantity(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

def small_to_large_mapping(small_pixel, large_dimensions):
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
    return large_x, large_y

def score_tile(np_tile, tissue_percent, slide_num, row, col):
    color_factor = hsv_purple_pink_factor(np_tile)
    s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
    amount = tissue_quantity(tissue_percent)
    quantity_factor = tissue_quantity_factor(amount)
    combined_factor = color_factor * s_and_v_factor * quantity_factor
    score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
    return score, color_factor, s_and_v_factor, quantity_factor

class Tile:

    def __init__(self, tile_summary, slide_num, slide_filepath, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e,
                 o_r_s, o_r_e, o_c_s,
                 o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
        self.tile_summary = tile_summary
        self.slide_num = slide_num
        self.slide_filepath = slide_filepath
        self.np_scaled_tile = np_scaled_tile
        self.tile_num = tile_num
        self.r = r
        self.c = c
        self.r_s = r_s
        self.r_e = r_e
        self.c_s = c_s
        self.c_e = c_e
        self.o_r_s = o_r_s
        self.o_r_e = o_r_e
        self.o_c_s = o_c_s
        self.o_c_e = o_c_e
        self.tissue_percentage = t_p
        self.color_factor = color_factor
        self.s_and_v_factor = s_and_v_factor
        self.quantity_factor = quantity_factor
        self.score = score

    def __str__(self):
        return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def tissue_quantity(self):
        return tissue_quantity(self.tissue_percentage)

    def get_pil_tile(self):
        return tile_to_pil_tile(self)

    def get_np_tile(self):
        return tile_to_np_tile(self)

    def save_tile(self):
        save_display_tile(self, save=True, display=False)

    def display_tile(self):
        save_display_tile(self, save=False, display=True)

    def display_with_histograms(self):
        display_tile(self, rgb_histograms=True, hsv_histograms=True)

    def get_np_scaled_tile(self):
        return self.np_scaled_tile

    def get_pil_scaled_tile(self):
        return np_to_pil(self.np_scaled_tile)

def get_tile_data_path(slide_number, slide_filepath):

    if not os.path.exists(TILE_DATA_DIR):
        os.makedirs(TILE_DATA_DIR)
    file_path = os.path.join(TILE_DATA_DIR, get_tile_data_filename(slide_number, slide_filepath))
    return file_path

def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):

    r = row_tile_size * num_row_tiles + title_area_height
    c = col_tile_size * num_col_tiles
    summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
    # add gray edges so that tile text does not get cut off
    summary_img.fill(120)
    # color title area white
    summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
    summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
    summary = np_to_pil(summary_img)
    return summary

def tile_border_color(tissue_percentage):

    if tissue_percentage >= TISSUE_HIGH_THRESH:
        border_color = HIGH_COLOR
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        border_color = MEDIUM_COLOR
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        border_color = LOW_COLOR
    else:
        border_color = NONE_COLOR
    return border_color

def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
    for x in range(0, border_size):
        draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)

def save_tile_summary_image(pil_img, slide_num, slide_filepath):
    t = Time()
    filepath = get_tile_summary_image_path(slide_num, slide_filepath)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum", str(t.elapsed()), filepath))

def save_tile_summary_on_original_image(pil_img, slide_num, slide_filepath):
    t = Time()
    filepath = get_tile_summary_on_original_image_path(slide_num, slide_filepath)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig", str(t.elapsed()), filepath))

def faded_tile_border_color(tissue_percentage):
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        border_color = FADED_THRESH_COLOR
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        border_color = FADED_MEDIUM_COLOR
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        border_color = FADED_LOW_COLOR
    else:
        border_color = FADED_NONE_COLOR
    return border_color

def add_tile_stats_to_top_tile_summary(pil_img, tiles, z):
    np_sum = pil_to_np_rgb(pil_img)
    sum_r, sum_c, sum_ch = np_sum.shape
    np_stats = np_tile_stat_img(tiles)
    st_r, st_c, _ = np_stats.shape
    combo_c = sum_c + st_c
    combo_r = max(sum_r, st_r + z)
    combo = np.zeros([combo_r, combo_c, sum_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:sum_r, 0:sum_c] = np_sum
    combo[z:st_r + z, sum_c:sum_c + st_c] = np_stats
    result = np_to_pil(combo)
    return result

def save_good_tiles_image(pil_img, slide_num, slide_filepath):
    t = Time()
    filepath = get_good_tiles_image_path(slide_num, slide_filepath)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Good Tiles Image", str(t.elapsed()), filepath))

def save_good_tiles_on_original_image(pil_img, slide_num, slide_filepath):
    t = Time()
    filepath = get_good_tiles_on_original_image_path(slide_num, slide_filepath)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Good Orig", str(t.elapsed()), filepath))

def hsv_purple_pink_factor(rgb):
    hues = rgb_to_hues(rgb)
    hues = hues[hues >= 260]  # exclude hues under 260
    hues = hues[hues <= 340]  # exclude hues over 340
    if len(hues) == 0:
        return 0  # if no hues between 260 and 340, then not purple or pink
    pu_dev = hsv_purple_deviation(hues)
    pi_dev = hsv_pink_deviation(hues)
    avg_factor = (340 - np.average(hues)) ** 2

    if pu_dev == 0:  # avoid divide by zero if tile has no tissue
        return 0

    factor = pi_dev / pu_dev * avg_factor
    return factor

def hsv_saturation_and_value_factor(rgb):
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    s = filter_hsv_to_s(hsv)
    v = filter_hsv_to_v(hsv)
    s_std = np.std(s)
    v_std = np.std(v)
    if s_std < 0.05 and v_std < 0.05:
        factor = 0.4
    elif s_std < 0.05:
        factor = 0.7
    elif v_std < 0.05:
        factor = 0.7
    else:
        factor = 1

    factor = factor ** 2
    return factor

def tissue_quantity_factor(amount):
    if amount == TissueQuantity.HIGH:
        quantity_factor = 1.0
    elif amount == TissueQuantity.MEDIUM:
        quantity_factor = 0.2
    elif amount == TissueQuantity.LOW:
        quantity_factor = 0.1
    else:
        quantity_factor = 0.0
    return quantity_factor

def get_tile_summary_image_path(slide_number, slide_filepath):
    if not os.path.exists(TILE_SUMMARY_DIR):
        os.makedirs(TILE_SUMMARY_DIR)
    img_path = os.path.join(TILE_SUMMARY_DIR, get_tile_summary_image_filename(slide_number, slide_filepath))
    return img_path

def get_tile_summary_on_original_image_path(slide_number, slide_filepath):
    if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_DIR):
        os.makedirs(TILE_SUMMARY_ON_ORIGINAL_DIR)
    img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_DIR, get_tile_summary_image_filename(slide_number, slide_filepath))
    return img_path

def np_tile_stat_img(tiles):
    tt = sorted(tiles, key=lambda t: (t.r, t.c), reverse=False)
    tile_stats = "Tile Score Statistics:\n"
    count = 0
    for t in tt:
        if count > 0:
            tile_stats += "\n"
        count += 1
        tup = (t.r, t.c, t.rank, t.tissue_percentage, t.color_factor, t.s_and_v_factor, t.quantity_factor, t.score)
        tile_stats += "R%03d C%03d #%003d TP:%6.2f%% CF:%4.0f SVF:%4.2f QF:%4.2f S:%0.4f" % tup
    np_stats = np_text(tile_stats, font_path=SUMMARY_TITLE_FONT_PATH, font_size=14)
    return np_stats

def get_good_tiles_image_path(slide_number, slide_filepath):
    if not os.path.exists(GOOD_TILES_DIR):
        os.makedirs(GOOD_TILES_DIR)
    img_path = os.path.join(GOOD_TILES_DIR, get_good_tiles_image_filename(slide_number, slide_filepath))
    return img_path

def get_good_tiles_on_original_image_path(slide_number, slide_filepath):
    if not os.path.exists(GOOD_TILES_ON_ORIGINAL_DIR):
        os.makedirs(GOOD_TILES_ON_ORIGINAL_DIR)
    img_path = os.path.join(GOOD_TILES_ON_ORIGINAL_DIR, get_good_tiles_image_filename(slide_number, slide_filepath))
    return img_path

def rgb_to_hues(rgb):
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    h = filter_hsv_to_h(hsv, display_np_info=False)
    return h

def hsv_purple_deviation(hsv_hues):
    purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
    return purple_deviation

def hsv_pink_deviation(hsv_hues):
    pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
    return pink_deviation

def get_tile_summary_image_filename(slide_number, slide_filepath, thumbnail=False):
    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT
    padded_sl_num = str(slide_number).zfill(3)
    slide_filepath = slide_filepath.replace("/", "\\")
    slide_name = slide_filepath.split("\\")[-1][:-4]

    training_img_path = get_training_image_path(slide_number, slide_filepath)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + slide_name + "-" + str(large_w) + "x" + str(
        large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_SUMMARY_SUFFIX + "." + ext

    return img_filename

def np_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
            font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
    pil_img = pil_text(text, w_border, h_border, font_path, font_size,
                       text_color, background)
    np_img = pil_to_np_rgb(pil_img)
    return np_img

def get_good_tiles_image_filename(slide_number, slide_filepath, thumbnail=False):
    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT
    padded_sl_num = str(slide_number).zfill(3)
    slide_filepath = slide_filepath.replace("/", "\\")
    slide_name = slide_filepath.split("\\")[-1][:-4]

    training_img_path = get_training_image_path(slide_number, slide_filepath)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + slide_name + "-" + str(large_w) + "x" + str(
        large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + GOOD_TILES_SUFFIX + "." + ext

    return img_filename


def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):

    font = ImageFont.truetype(font_path, font_size)
    font = ImageFont.load_default()
    x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
    image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
    draw = ImageDraw.Draw(image)
    draw.text((w_border, h_border), text, text_color, font=font)
    return image

def get_tile_data_filename(slide_number, slide_filepath):
    padded_sl_num = str(slide_number).zfill(3)
    slide_filepath = slide_filepath.replace("/", "\\")
    slide_name = slide_filepath.split("\\")[-1][:-4]

    training_img_path = get_training_image_path(slide_number, slide_filepath)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    data_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + slide_name + "-" + str(large_w) + "x" + str(
        large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_DATA_SUFFIX + ".csv"

    return data_filename

def singleprocess_training_slides_to_images():
    t = Time()

    num_train_images = get_num_training_slides()
    training_slide_range_to_images(1, num_train_images)

    t.elapsed_display()

def singleprocess_apply_filters_to_images(save=True, display=False, html=False, image_num_list=None):

    t = Time()
    print("Applying filters to images\n")

    if image_num_list is not None:
        _, info = apply_filters_to_image_list(image_num_list, save, display)
    else:
        num_training_slides = get_num_training_slides()
        (s, e, info) = apply_filters_to_image_range(1, num_training_slides, save, display)

    print("Time to apply filters to all images: %s\n" % str(t.elapsed()))

    if html:
        pass

def singleprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True, save_top_tiles=False,
                                           save_good_tiles=True, html=False, image_num_list=None, slide_filepath=None):
    t = Time()
    print("Generating tile summaries\n")

    if image_num_list is not None:
        image_num_list, tile_summaries_dict = image_list_to_tiles(image_num_list, slide_filepath, display, save_summary,
                                                                  save_data, save_top_tiles, save_good_tiles)
    else:
        num_training_slides = get_num_training_slides()
        image_num_list, tile_summaries_dict = image_range_to_tiles(1, num_training_slides, display, save_summary,
                                                                   save_data, save_top_tiles, save_good_tiles)

    print("Time to generate tile summaries: %s\n" % str(t.elapsed()))

    if html:
        pass

def multiprocess_training_slides_to_images():
    timer = Time()

    # how many processes to use
    num_processes = multiprocessing.cpu_count() - 4
    pool = multiprocessing.Pool(num_processes)

    num_train_images = get_num_training_slides()
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    print("Number of processes: " + str(num_processes))
    print("Number of training images: " + str(num_train_images))

    # each task specifies a range of slides
    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((start_index, end_index))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(training_slide_range_to_images, t))

    for result in results:
        (start_ind, end_ind) = result.get()
        if start_ind == end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind, end_ind))

    timer.elapsed_display()

def multiprocess_apply_filters_to_images(save=True, display=False, html=False, image_num_list=None):
    timer = Time()
    print("Applying filters to images (multiprocess)\n")

    if save and not os.path.exists(FILTER_DIR):
        os.makedirs(FILTER_DIR)

    # how many processes to use
    num_processes = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(num_processes)

    if image_num_list is not None:
        num_train_images = len(image_num_list)
    else:
        num_train_images = get_num_training_slides()
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    print("Number of processes: " + str(num_processes))
    print("Number of training images: " + str(num_train_images))

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        if image_num_list is not None:
            sublist = image_num_list[start_index - 1:end_index]
            tasks.append((sublist, save, display))
            print("Task #" + str(num_process) + ": Process slides " + str(sublist))
        else:
            tasks.append((start_index, end_index, save, display))
            if start_index == end_index:
                print("Task #" + str(num_process) + ": Process slide " + str(start_index))
            else:
                print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        if image_num_list is not None:
            results.append(pool.apply_async(apply_filters_to_image_list, t))
        else:
            results.append(pool.apply_async(apply_filters_to_image_range, t))

    html_page_info = dict()
    for result in results:
        if image_num_list is not None:
            (image_nums, html_page_info_res) = result.get()
            html_page_info.update(html_page_info_res)
            print("Done filtering slides: %s" % image_nums)
        else:
            (start_ind, end_ind, html_page_info_res) = result.get()
            html_page_info.update(html_page_info_res)
            if (start_ind == end_ind):
                print("Done filtering slide %d" % start_ind)
            else:
                print("Done filtering slides %d through %d" % (start_ind, end_ind))

    if html:
        pass

    print("Time to apply filters to all images (multiprocess): %s\n" % str(timer.elapsed()))

def multiprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=False, save_top_tiles=False,
                                          save_good_tiles=True, html=False, image_num_list=None):
    timer = Time()
    print("Generating tile summaries (multiprocess)\n")

    if save_summary and not os.path.exists(TILE_SUMMARY_DIR):
        os.makedirs(TILE_SUMMARY_DIR)

    # how many processes to use
    num_processes = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(num_processes)

    if image_num_list is not None:
        num_train_images = len(image_num_list)
    else:
        num_train_images = get_num_training_slides()
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes
 


    print("Number of processes: " + str(num_processes))
    print("Number of training images: " + str(num_train_images))

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        if image_num_list is not None:
            sublist = image_num_list[start_index - 1:end_index]
            tasks.append((sublist, display, save_summary, save_data, save_top_tiles, save_good_tiles))
            print("Task #" + str(num_process) + ": Process slides " + str(sublist))
        else:
            tasks.append((start_index, end_index, display, save_summary, save_data, save_top_tiles, save_good_tiles))
            if start_index == end_index:
                print("Task #" + str(num_process) + ": Process slide " + str(start_index))
            else:
                print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        if image_num_list is not None:
            results.append(pool.apply_async(image_list_to_tiles, t))
        else:
            results.append(pool.apply_async(image_range_to_tiles, t))

    slide_nums = list()
    tile_summaries_dict = dict()
    for result in results:
        image_nums, tile_summaries = result.get()
        slide_nums.extend(image_nums)
        tile_summaries_dict.update(tile_summaries)
        print("Done tiling slides: %s" % image_nums)

    if html:
        pass

    print("Time to generate tile previews (multiprocess): %s\n" % str(timer.elapsed()))

def slide_info(display_all_properties=False):
    """
  Display information (such as properties) about training images.
    展示训练图像的信息
  Args:
    display_all_properties: If True, display all available slide properties.
  """
    t = Time()

    num_train_images = get_num_training_slides()
    obj_pow_20_list = []
    obj_pow_20_list_1 = []
    obj_pow_40_list = []
    obj_pow_other_list = []
    for slide_num in range(1, num_train_images + 1):
        slide_filepath = os.path.join(SRC_TRAIN_DIR, os.listdir(SRC_TRAIN_DIR)[slide_num - 1])
        print("\nOpening Slide #%d: %s" % (slide_num, slide_filepath))
        print(slide_filepath.split("\\")[-1])
        slide = open_slide(slide_filepath)
        print("Level count: %d" % slide.level_count)
        print("Level dimensions: " + str(slide.level_dimensions))
        print("Level downsamples: " + str(slide.level_downsamples))
        print("Dimensions: " + str(slide.dimensions))
        objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        print("Objective power: " + str(objective_power))
        if objective_power == 20:
            obj_pow_20_list.append(slide_num)
            obj_pow_20_list_1.append(slide_filepath.split("\\")[-1])
        elif objective_power == 40:
            obj_pow_40_list.append(slide_num)
        else:
            obj_pow_other_list.append(slide_num)
        print("Associated images:")
        for ai_key in slide.associated_images.keys():
            print("  " + str(ai_key) + ": " + str(slide.associated_images.get(ai_key)))
        print("Format: " + str(slide.detect_format(slide_filepath)))
        if display_all_properties:
            print("Properties:")
            for prop_key in slide.properties.keys():
                print("  Property: " + str(prop_key) + ", value: " + str(slide.properties.get(prop_key)))

    print("\n\nSlide Magnifications:")
    print("  20x Slides: " + str(obj_pow_20_list))
    print("  20x Slides: " + str(obj_pow_20_list_1))
    print("  40x Slides: " + str(obj_pow_40_list))
    print("  ??x Slides: " + str(obj_pow_other_list) + "\n")

    t.elapsed_display()
