import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os


def split_dataset(input_path, test_size, seed):
    global train_output, test_output

    # 读取输入文件
    data = pd.read_csv(input_path)

    # 随机分割数据集
    train, test = train_test_split(data, test_size=test_size, random_state=seed, stratify=data['label'])

    # 为输出文件创建名称
    base_name, ext = os.path.splitext(input_path)
    train_output = base_name + "_train_val" + ext
    test_output = base_name + "_test" + ext

    # 保存训练集和测试集
    train.to_csv(train_output, index=False)
    test.to_csv(test_output, index=False)

def save_cross_val_splits_0(train_val_data, test_data, output_dir_path):
    train_data, val_data = train_val_data, test_data
    result = pd.DataFrame(index=range(len(train_val_data)), columns=["train", "val", "test"])
    result.loc[train_data.index, "train"] = train_data["slide_id"].astype(str)
    result.loc[val_data.index, "val"] = val_data["slide_id"].astype(str)
    result.loc[test_data.index, "test"] = test_data["slide_id"].astype(str)
    result.to_csv(os.path.join(output_dir_path, "splits_0.csv"), index_label="")

    train_val_data = pd.read_csv(os.path.join(output_dir_path, "splits_0.csv"), index_col=0)
    train = train_val_data['train'].dropna().reset_index(drop=True)
    val = train_val_data['val'].dropna().reset_index(drop=True)
    test = train_val_data['test'].dropna().reset_index(drop=True)
    new_data = pd.DataFrame(columns=['train', 'val', 'test'])

    # 根据train和val中的最大行数创建新的DataFrame
    max_rows = max(len(train), len(val))
    for i in range(max_rows):
        train_value = train[i] if i < len(train) else None
        val_value = val[i] if i < len(val) else None
        test_value = test[i] if i < len(test) else None
        new_row = pd.DataFrame({'train': [train_value], 'val': [val_value], 'test': [test_value]})
        new_data = pd.concat([new_data, new_row], ignore_index=True)

    new_data.to_csv(os.path.join(output_dir_path, "splits_0.csv"), index_label="")

def generate_bool_csv_0(output_dir_path):
    data = pd.read_csv(os.path.join(output_dir_path, "splits_0.csv"), index_col=0)
    all_samples = pd.concat([data["train"].dropna(), data["val"].dropna(), data["test"].dropna()], ignore_index=True)
    result = pd.DataFrame(index=all_samples, columns=["train", "val", "test"])
    result.loc[data["train"].dropna(), "train"] = True
    result.loc[data["train"].dropna(), "val"] = False
    result.loc[data["train"].dropna(), "test"] = False
    result.loc[data["val"].dropna(), "train"] = False
    result.loc[data["val"].dropna(), "val"] = True
    # result.loc[data["val"].dropna(), "test"] = False
    result.loc[data["test"].dropna(), "train"] = False
    # result.loc[data["test"].dropna(), "val"] = False
    result.loc[data["test"].dropna(), "test"] = True

    result = result.astype(bool)
    result.to_csv(os.path.join(output_dir_path, "splits_0_bool.csv"), index_label="")

def generate_descriptor_csv_0(train_val_dataset_csv_path, test_dataset_csv_path, output_dir_path):
    train_val_data = pd.read_csv(train_val_dataset_csv_path)
    test_data = pd.read_csv(test_dataset_csv_path)
    split_data = pd.read_csv(os.path.join(output_dir_path, "splits_0.csv"), index_col=0)
    train_samples, val_samples, test_samples = split_data["train"].dropna(), split_data["val"].dropna(), split_data[
        "test"].dropna()
    train_val_sample_ids = train_val_data["slide_id"].astype(str)
    test_sample_ids = test_data["slide_id"].astype(str)
    train_labels, _ = train_val_data[train_val_sample_ids.isin(train_samples)]["label"], train_val_data[train_val_sample_ids.isin(val_samples)]["label"]
    val_labels = test_data[test_sample_ids.isin(test_samples)]["label"]
    test_labels = test_data[test_sample_ids.isin(test_samples)]["label"]
    result = pd.DataFrame(columns=["train", "val", "test"])
    result["train"], result["val"], result["test"] = train_labels.value_counts(), val_labels.value_counts(), test_labels.value_counts()
    result.to_csv(os.path.join(output_dir_path, "splits_0_descriptor.csv"), index_label="")



def save_cross_val_splits(train_val_data, test_data, fold, train_index, val_index, output_dir_path):
    train_data, val_data = train_val_data.iloc[train_index], train_val_data.iloc[val_index]
    result = pd.DataFrame(index=range(len(train_val_data)), columns=["train", "val", "test"])
    result.loc[train_data.index, "train"] = train_data["slide_id"].astype(str)
    result.loc[val_data.index, "val"] = val_data["slide_id"].astype(str)
    result.loc[test_data.index, "test"] = test_data["slide_id"].astype(str)
    result.to_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), index_label="")

    train_val_data = pd.read_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), index_col=0)
    train = train_val_data['train'].dropna().reset_index(drop=True)
    val = train_val_data['val'].dropna().reset_index(drop=True)
    test = train_val_data['test'].dropna().reset_index(drop=True)
    new_data = pd.DataFrame(columns=['train', 'val', 'test'])

    # 根据train和val中的最大行数创建新的DataFrame
    max_rows = max(len(train), len(val))
    for i in range(max_rows):
        train_value = train[i] if i < len(train) else None
        val_value = val[i] if i < len(val) else None
        test_value = test[i] if i < len(test) else None
        new_row = pd.DataFrame({'train': [train_value], 'val': [val_value], 'test': [test_value]})
        new_data = pd.concat([new_data, new_row], ignore_index=True)

    new_data.to_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), index_label="")


def generate_bool_csv(fold, output_dir_path):
    data = pd.read_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), index_col=0)
    all_samples = pd.concat([data["train"].dropna(), data["val"].dropna(), data["test"].dropna()], ignore_index=True)
    result = pd.DataFrame(index=all_samples, columns=["train", "val", "test"])
    result.loc[data["train"].dropna(), "train"] = True
    result.loc[data["train"].dropna(), "val"] = False
    result.loc[data["train"].dropna(), "test"] = False
    result.loc[data["val"].dropna(), "train"] = False
    result.loc[data["val"].dropna(), "val"] = True
    result.loc[data["val"].dropna(), "test"] = False
    result.loc[data["test"].dropna(), "train"] = False
    result.loc[data["test"].dropna(), "val"] = False
    result.loc[data["test"].dropna(), "test"] = True

    result = result.astype(bool)
    result.to_csv(os.path.join(output_dir_path, f"splits_{fold}_bool.csv"), index_label="")


def generate_descriptor_csv(fold, train_val_dataset_csv_path, test_dataset_csv_path, output_dir_path):
    train_val_data = pd.read_csv(train_val_dataset_csv_path)
    test_data = pd.read_csv(test_dataset_csv_path)
    split_data = pd.read_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), index_col=0)
    train_samples, val_samples, test_samples = split_data["train"].dropna(), split_data["val"].dropna(), split_data[
        "test"].dropna()
    train_val_sample_ids = train_val_data["slide_id"].astype(str)
    test_sample_ids = test_data["slide_id"].astype(str)
    train_labels, val_labels = train_val_data[train_val_sample_ids.isin(train_samples)]["label"], train_val_data[train_val_sample_ids.isin(val_samples)]["label"]
    test_labels = test_data[test_sample_ids.isin(test_samples)]["label"]
    result = pd.DataFrame(columns=["train", "val", "test"])
    result["train"], result["val"], result["test"] = train_labels.value_counts(), val_labels.value_counts(), test_labels.value_counts()
    result.to_csv(os.path.join(output_dir_path, f"splits_{fold}_descriptor.csv"), index_label="")


def main(train_val_dataset_csv_path, test_data_csv_path, input_dir_path, output_dir_path, seed, kfold):
    output_dir_path = output_dir_path + os.path.basename(input_dir_path).split(".csv")[0] + "_100" + "_train_val_test"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    train_val_data = pd.read_csv(train_val_dataset_csv_path)
    test_data = pd.read_csv(test_data_csv_path)
    if kfold > 1:
        kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)

        for fold, (train_index, val_index) in enumerate(kf.split(train_val_data, train_val_data["label"])):
            save_cross_val_splits(train_val_data, test_data, fold, train_index, val_index, output_dir_path)
            generate_bool_csv(fold, output_dir_path)
            generate_descriptor_csv(fold, train_val_dataset_csv_path, test_data_csv_path, output_dir_path)

    elif kfold == 1:

        save_cross_val_splits_0(train_val_data, test_data, output_dir_path)
        generate_bool_csv_0(output_dir_path)
        generate_descriptor_csv_0(train_val_dataset_csv_path, test_data_csv_path, output_dir_path)



    else:
        raise ValueError("kfold must be >= 1")


def float_or_int(value):
    try:
        return int(value)
    except ValueError:
        return float(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val and test subsets.请注意，这里的划分方式是先独立地划分出测试集，然后再对训练集和验证集进行分层k折划分")
    parser.add_argument("--input_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--test_size", type=float_or_int, default=0.1, help="Size of the test set.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for shuffling (default: 1).")
    parser.add_argument("--kfold", type=int, default=10, help="Number of folds for cross-validation (default: 10).")
    parser.add_argument("--save_sub_dataset_csv", action="store_true", default=False)
    args = parser.parse_args()
    split_dataset(args.input_path, args.test_size, args.seed)
    print("原始数据集已在当前路径下生成train_val和test两个csv文件，分别存放训练验证集和测试集")
    print(f"已在splits/{os.path.basename(args.input_path) + '_100'}下生成对应的文件")
    main(train_output, test_output, args.input_path, "splits/", args.seed, args.kfold)
    if not args.save_sub_dataset_csv:
        os.remove(train_output)
        os.remove(test_output)
        print("生成的两个训练验证集和测试集文件已被删除")
