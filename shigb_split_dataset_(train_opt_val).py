import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os


def split_train_opt_dataset(train_opt_data, opt_size, seed):
	train, opt = train_test_split(train_opt_data, test_size=opt_size, random_state=seed,
								  stratify=train_opt_data['label'])
	return train, opt


def save_cross_val_splits(train_opt_eval_data, fold, train_opt_index, eval_index, output_dir_path):
	train_opt_data, eval_data = train_opt_eval_data.iloc[train_opt_index], train_opt_eval_data.iloc[eval_index]
	train_data, opt_data = split_train_opt_dataset(train_opt_data, 0.25, 728)
	result = pd.DataFrame(index=range(len(train_opt_eval_data)), columns=["train", "val", "test"])
	result.loc[train_data.index, "train"] = train_data["slide_id"].astype(str)
	result.loc[opt_data.index, "val"] = opt_data["slide_id"].astype(str)
	result.loc[eval_data.index, "test"] = eval_data["slide_id"].astype(str)
	result.to_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), encoding="utf-8-sig", index_label="")

	train_opt_eval_data = pd.read_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), index_col=0)
	train_data = train_opt_eval_data['train'].dropna().reset_index(drop=True)
	opt_data = train_opt_eval_data['val'].dropna().reset_index(drop=True)
	eval_data = train_opt_eval_data['test'].dropna().reset_index(drop=True)
	new_data = pd.DataFrame(columns=['train', 'val', 'test'])

	max_rows = max(len(train_data), len(opt_data), len(eval_data))
	for i in range(max_rows):
		train_value = train_data[i] if i < len(train_data) else None
		opt_value = opt_data[i] if i < len(opt_data) else None
		eval_value = eval_data[i] if i < len(eval_data) else None
		new_row = pd.DataFrame({'train': [train_value], 'val': [opt_value], 'test': [eval_value]})
		new_data = pd.concat([new_data, new_row], ignore_index=True)

	print(new_data)
	new_data.to_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), encoding="utf-8-sig", index_label="")


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
	print(result)
	result.to_csv(os.path.join(output_dir_path, f"splits_{fold}_bool.csv"), encoding="utf-8-sig", index_label="")


def generate_descriptor_csv(fold, original_data, output_dir_path):
	split_data = pd.read_csv(os.path.join(output_dir_path, f"splits_{fold}.csv"), index_col=0)
	train_samples = split_data["train"].dropna()
	opt_samples = split_data["val"].dropna()
	eval_samples = split_data["test"].dropna()
	samples_id = original_data["slide_id"].astype(str)
	train_labels, opt_labels, eval_labels = original_data[samples_id.isin(train_samples)]["label"], original_data[samples_id.isin(opt_samples)]["label"], original_data[samples_id.isin(eval_samples)]["label"]
	result = pd.DataFrame(columns=["train", "val", "test"])
	result["train"], result["val"], result["test"] = train_labels.value_counts(), opt_labels.value_counts(), eval_labels.value_counts()

	print(result)
	result.to_csv(os.path.join(output_dir_path, f"splits_{fold}_descriptor.csv"), index_label="")


def main(input_dir_path, seed, kfold):
	output_dir_path = "splits/"+os.path.basename(input_dir_path).split(".csv")[0] + "_100" + "_train_val_test"
	if not os.path.exists(output_dir_path):
		os.makedirs(output_dir_path)
	train_opt_eval_data = pd.read_csv(input_dir_path)
	kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
	for fold, (train_opt_index, eval_index) in enumerate(kf.split(train_opt_eval_data, train_opt_eval_data["label"])):
		save_cross_val_splits(train_opt_eval_data, fold, train_opt_index, eval_index, output_dir_path)
		generate_bool_csv(fold, output_dir_path)
		generate_descriptor_csv(fold, train_opt_eval_data, output_dir_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Split dataset into train/val and test subsets. 请注意, 这里的划分方式是对训练优化集和评估集先进行分层k折划分, 然后对训练优化集进行一般的分层划分")
	parser.add_argument("--input_path", type=str, help="Path to the input CSV file.")
	parser.add_argument("--seed", type=int, default=728, help="Random seed for shuffling (default: 1).")
	parser.add_argument("--kfold", type=int, default=5, help="Number of folds for cross-validation (default: 5).")

	args = parser.parse_args()
	main(args.input_path, args.seed, args.kfold)
