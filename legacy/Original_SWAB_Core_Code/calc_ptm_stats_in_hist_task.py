import open_clip
import yaml
import torch
from data.get_dataset import get_dataset
from torchvision import transforms
from tqdm import tqdm
import os
from model.blip_class import BLIP
from model.beit_class import BEIT3
import pickle
import torch.nn.functional as F
import json


# --- 新增部分：放在 import 语句之后，其他函数之前 ---

def calculate_ece(probs, labels, n_bins=15):
    """
    计算 Expected Calibration Error (ECE)
    probs: [n_samples, n_classes] 经过 softmax 后的概率
    labels: [n_samples] 真实标签
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # 落在当前 bin 内的样本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def calculate_ace(probs, labels, n_bins=15):
    """
    计算 Adaptive Calibration Error (ACE) - 每个 bin 样本数相同
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    # 按置信度排序
    sorted_confidences, sorted_indices = torch.sort(confidences)
    sorted_accuracies = accuracies[sorted_indices]

    ace = 0.0
    N = len(probs)
    # 计算每个 bin 的大小
    bin_size = int(N / n_bins)

    for i in range(n_bins):
        start_idx = i * bin_size
        # 最后一个 bin 包含剩余所有样本
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else N

        if start_idx >= end_idx:
            break

        bin_confidences = sorted_confidences[start_idx:end_idx]
        bin_accuracies = sorted_accuracies[start_idx:end_idx]

        avg_confidence_in_bin = bin_confidences.mean()
        accuracy_in_bin = bin_accuracies.float().mean()

        ace += torch.abs(avg_confidence_in_bin - accuracy_in_bin)

    return (ace / n_bins).item()

def get_model_names():
    with open("../model/models.yml", 'r') as file:
        model_names = yaml.safe_load(file)

    model_names = [tuple(m) for m in model_names]

    return model_names

# def get_dataset_and_class_names():
#     with open("data/datasets_name.txt", 'r') as f:
#         dataset_names = [line.rstrip('\n') for line in f.readlines()]
#
#     dataset_dict = {}
#     for d in dataset_names:
#         _, test_dataset, _ = get_dataset(d)
#         dataset_dict[d] = test_dataset
#
#     class_name_dict = {}
#     for dataset in dataset_names:
#         class_name_file = f"data/datasets/classnames/{dataset}.txt"
#
#         with open(class_name_file, 'r') as file:
#             class_name_list = [line.strip() for line in file]
#
#         class_name_dict[dataset] = class_name_list
#
#     return dataset_names, dataset_dict, class_name_dict
# 修改 calc_ptm_stats_in_hist_task.py
# 替换原本的 get_dataset_and_class_names 函数

def get_dataset_and_class_names():
    with open("../data/datasets_name.txt", 'r') as f:
        # 读取所有数据集名称
        all_dataset_names = [line.rstrip('\n') for line in f.readlines()]

    dataset_names = []
    dataset_dict = {}

    print("\n[检查数据集完整性]...")
    for d in all_dataset_names:
        try:
            # 尝试加载数据集
            # 如果文件夹为空，get_dataset 内部调用 ImageFolder 时会抛出错误
            _, test_dataset, _ = get_dataset(d)

            # 额外检查：如果数据集长度为0，也视为无效
            if len(test_dataset) > 0:
                dataset_dict[d] = test_dataset
                dataset_names.append(d)
                print(f"  [OK] {d:<20} (Images: {len(test_dataset)})")
            else:
                print(f"  [SKIP] {d:<20} (Dataset is empty)")

        except Exception as e:
            # 捕获 FileNotFoundError 或其他加载错误
            # 这里的 e 就是你刚才遇到的 "Couldn't find any class folder..."
            print(f"  [SKIP] {d:<20} (Raw data missing)")
            continue

    class_name_dict = {}
    for dataset in dataset_names:
        # 只读取加载成功的数据集的类别名
        class_name_file = f"data/datasets/classnames/{dataset}.txt"
        with open(class_name_file, 'r') as file:
            class_name_list = [line.strip() for line in file]
        class_name_dict[dataset] = class_name_list

    print(f"\n[Summary] 成功加载 {len(dataset_names)} 个数据集。")
    print(f"即将开始计算: {dataset_names}\n")

    return dataset_names, dataset_dict, class_name_dict

def run_one_model_for_img_feat(model, model_name, dataset_names, dataset_dict, class_name_dict, preprocess):
    img_feat_dict = {}

    for d in tqdm(dataset_names,  desc=f'{model_name[0]}, {model_name[1]}', leave=False):
        dataset = dataset_dict[d]

        if d in {"fer2013","mnist"}:
            preprocess_list = [transforms.Grayscale(num_output_channels=3)]+list(preprocess.transforms)
            dataset.transform = transforms.Compose(preprocess_list)

        else:
            dataset.transform = preprocess

        # get dataloader
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                num_workers=8,
                pin_memory=True,
                shuffle=False,
            )

        with torch.no_grad():
            data = {}
            for images, labels in tqdm(dataloader, desc='Dataloader', leave=False):
                images = images.cuda()
                labels = labels.cuda()

                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                for feature, label in zip(image_features.detach().cpu(), labels):
                    class_name = class_name_dict[d][label.item()]
                    if class_name not in data:
                        data[class_name] = []
                    data[class_name].append(feature)

            for class_name, feat_list in data.items():
                data[class_name] = torch.stack(feat_list)

        img_feat_dict[d] = data

    return img_feat_dict

def calculate_text_classifier_weights(model, classnames, templates, tokenizer=None, model_name=None):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, leave=False):
            texts = [template.replace('{c}', classname) for template in templates]

            text_feat_list = []
            if "BEIT" in model_name:
                for text in texts:
                    text_features = model.encode_text(text)
                    text_feat_list.append(text_features)
                class_embeddings = torch.cat(text_feat_list, dim=0)
            elif "BLIP" in model_name:
                for text in texts:
                    text_features = model.encode_text(text)
                    text_feat_list.append(text_features)
                class_embeddings = torch.cat(text_feat_list, dim=0)
            else:
                texts = tokenizer(texts).cuda()
                class_embeddings = model.encode_text(texts)

            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def calculate_captions_text_feat(model, datasets, tokenizer=None, model_name=None):
    with torch.no_grad():
        captions_feat = {}
        for classname, captions in tqdm(datasets.items()):
            text_feat_list = []
            try:
                if "BEIT" in model_name or "BLIP" in model_name:
                    for text in captions:
                        text_features = model.encode_text(text)
                        text_feat_list.append(text_features)
                    text_features = torch.cat(text_feat_list, dim=0)
                else:
                    texts = tokenizer(captions).cuda()
                    text_features = model.encode_text(texts)

                text_features /= text_features.norm(dim=-1, keepdim=True)

                captions_feat[classname] = text_features

            except:
                print(captions)
                assert 0

    return captions_feat

def calculate_syn_text_feat(model, datasets, tokenizer=None, model_name=None):
    with torch.no_grad():
        syns_feat = {}
        for classname, syns in tqdm(datasets.items()):
            text_feat_list = []
            try:
                if "BEIT" in model_name or "BLIP" in model_name:
                    for text in syns:
                        text_features = model.encode_text(text)
                        text_feat_list.append(text_features)
                    text_features = torch.cat(text_feat_list, dim=0)
                else:
                    texts = tokenizer(syns).cuda()
                    text_features = model.encode_text(texts)

                text_features /= text_features.norm(dim=-1, keepdim=True)

                syns_feat[classname] = text_features

            except:
                print(syns)
                assert 0

    return syns_feat

def run_one_model_for_text_feat(model, model_name, dataset_names, tokenizer):
    text_classifier_dict = {}
    caption_text_feat_dict = {}
    syn_text_feat_dict = {}

    caption_datasets = {}
    syn_datasets = {}
    for d in dataset_names:
        with open(f'data/captions_dataset/{d}.json', 'r') as f:
            caption_datasets[d] = json.load(f)
        with open(f'data/syn_dataset/{d}.json', 'r') as f:
            syn_datasets[d] = json.load(f)

    for dataset in tqdm(dataset_names, desc=str(model_name)):
        with open(f'LOVM/templates/{dataset}.txt', 'r') as f:
            templates = [line.rstrip('\n') for line in f.readlines()]

        with open(f'data/datasets/classnames/{dataset}.txt', 'r') as f:
            classes = [line.rstrip('\n') for line in f.readlines()]

        if "BEIT" in model_name[0] or "BLIP" in model_name[0]:
            text_classifier_dict[dataset] = calculate_text_classifier_weights(model, classes, templates, model_name=model_name[0])
            caption_text_feat_dict[dataset] = calculate_captions_text_feat(model, caption_datasets[dataset], model_name=model_name[0])
            syn_text_feat_dict[dataset] = calculate_syn_text_feat(model, syn_datasets[dataset], model_name=model_name[0])
        else:
            text_classifier_dict[dataset] = calculate_text_classifier_weights(model, classes, templates, tokenizer=tokenizer, model_name=model_name[0])
            caption_text_feat_dict[dataset] = calculate_captions_text_feat(model, caption_datasets[dataset], tokenizer=tokenizer, model_name=model_name[0])
            syn_text_feat_dict[dataset] = calculate_syn_text_feat(model, syn_datasets[dataset], model_name=model_name[0], tokenizer=tokenizer)

    return text_classifier_dict, caption_text_feat_dict, syn_text_feat_dict

def extract_img_feat_and_text_feat(model_names, dataset_names, dataset_dict, class_name_dict):
    for m in tqdm(model_names, desc="model"):
        model_name = "_".join(m)

        print(f"--------------- model {m} ---------------")
        if os.path.exists(f'ptm_stats_test/stats_on_hist_task/img_feat/{model_name}.pkl') and os.path.exists(f'ptm_stats_test/stats_on_hist_task/text_classifier/{model_name}.pkl') and os.path.exists(f'ptm_stats_test/stats_on_hist_task/caption_text_feat/{model_name}.pkl') and os.path.exists(f'ptm_stats_test/stats_on_hist_task/syn_text_feat/{model_name}.pkl'):
            continue
        else:
            if "BEIT" in m[0]:
                model_type, model_size, model_ptm_dataset = m[1].split("_")
                model = BEIT3(model_size=model_size, model_ptm_dataset=model_ptm_dataset)
                preprocess = model.transform
                model.model.to('cuda')
                model.model.eval()
                tokenizer = None
            elif "BLIP" in m[0]:
                model_type, model_size, model_ptm_dataset = m[1].split("_")
                model = BLIP(model_size=model_size, model_ptm_dataset=model_ptm_dataset)
                preprocess = model.transform
                model.model.to('cuda')
                model.model.eval()
                tokenizer = None
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(m[0], m[1], cache_dir='../model/checkpoint') # TODO
                model.to('cuda')
                model.eval()
                tokenizer = open_clip.get_tokenizer(m[0])

        img_feat_dict = run_one_model_for_img_feat(model, m, dataset_names, dataset_dict, class_name_dict, preprocess) # return dict

        with open(f'ptm_stats_test/stats_on_hist_task/img_feat/{model_name}.pkl', 'wb') as file:
            pickle.dump(img_feat_dict, file)

        text_classifier_dict, caption_text_feat_dict, syn_text_feat_dict = run_one_model_for_text_feat(model, m, dataset_names, tokenizer)

        with open(f'ptm_stats_test/stats_on_hist_task/text_classifier/{model_name}.pkl', 'wb') as file:
            pickle.dump(text_classifier_dict, file)

        with open(f'ptm_stats_test/stats_on_hist_task/caption_text_feat/{model_name}.pkl', 'wb') as file:
            pickle.dump(caption_text_feat_dict, file)

        with open(f'ptm_stats_test/stats_on_hist_task/syn_text_feat/{model_name}.pkl', 'wb') as file:
            pickle.dump(syn_text_feat_dict, file)

def cal_gap_dict(model_names, dataset_names, class_name_dict):
    for m in tqdm(model_names, desc="model"):
        model_name = "_".join(m)

        modality_gap_path = f"ptm_stats_test/stats_on_hist_task/modality_gap/{model_name}.pkl"
        f = open(modality_gap_path, 'wb')
        modality_gap_dict = {}

        img_feat_path = f"ptm_stats_test/stats_on_hist_task/img_feat/{model_name}.pkl"
        f1 = open(img_feat_path, 'rb')
        image_data_dict = pickle.load(f1)

        text_classifier_path = f"ptm_stats_test/stats_on_hist_task/text_classifier/{model_name}.pkl"
        f2 = open(text_classifier_path, 'rb')
        text_classifier_dict = pickle.load(f2)

        caption_text_feat_path = f'ptm_stats_test/stats_on_hist_task/caption_text_feat/{model_name}.pkl'
        f3 = open(caption_text_feat_path, 'rb')
        caption_text_data_dict = pickle.load(f3)

        for dataset_name in tqdm(dataset_names):
            modality_gap_dict[dataset_name] = {}
            total_modality_gap = []

            class_name_list = class_name_dict[dataset_name]

            img_data_list = []
            for _, img_data in image_data_dict[dataset_name].items():
                img_data_list.append(img_data)

            total_img_data = torch.cat(img_data_list, dim=0)

            caption_text_data_list = []
            for _, text_data in caption_text_data_dict[dataset_name].items():
                caption_text_data_list.append(text_data)

            total_caption_text_data = torch.cat(caption_text_data_list, dim=0)

            text_center = torch.mean(total_caption_text_data, dim=0)
            img_center = torch.mean(total_img_data, dim=0)

            text_std = torch.std(total_caption_text_data, dim=0)
            img_std = torch.std(total_img_data, dim=0)

            text_classifiers = text_classifier_dict[dataset_name]

            text_classifiers = ((text_classifiers.T.cuda() - text_center.cuda()) / text_std.cuda()).T

            for i, class_name in enumerate(class_name_list):
                img_feat = image_data_dict[dataset_name][class_name]
                img_feat = (img_feat - img_center) / img_std
                text_classifier = text_classifiers[:, i]
                modality_gaps = img_feat.cuda() - text_classifier.cuda()
                total_modality_gap.append(modality_gaps)
                modality_gap_dict[dataset_name][class_name] = torch.mean(modality_gaps, dim=0)

            modality_gap_dict[dataset_name]['dataset_mean'] = torch.mean(torch.cat(total_modality_gap, dim=0), dim=0)

        pickle.dump(modality_gap_dict, f)

def cal_model_acc(model_names, dataset_names, class_name_dict):
    logits_save_dir = "ptm_stats_test/logits"
    calib_save_dir = "ptm_stats_test/stats_on_hist_task/calibration_metrics"
    if not os.path.exists(calib_save_dir):
        os.makedirs(calib_save_dir)

    if not os.path.exists(logits_save_dir):
        os.makedirs(logits_save_dir)
    print(f"Logits will be saved to: {logits_save_dir}")

    for m in tqdm(model_names, desc="model"):
        model_performance_list = []

        model_name = "_".join(m)

        print("model_name:", model_name)

        acc_path = f"ptm_stats_test/stats_on_hist_task/class_level_acc/{model_name}.pkl"
        calib_path = f"{calib_save_dir}/{model_name}.pkl"

        f = open(acc_path, 'wb')
        acc_dict = {}
        calib_dict = {}

        img_feat_path = f"ptm_stats_test/stats_on_hist_task/img_feat/{model_name}.pkl"
        f1 = open(img_feat_path, 'rb')
        image_data_dict = pickle.load(f1)

        text_classifier_path = f"ptm_stats_test/stats_on_hist_task/text_classifier/{model_name}.pkl"
        f2 = open(text_classifier_path, 'rb')
        text_classifier_dict = pickle.load(f2)

        for dataset_name in dataset_names:
            total_top1_correct = 0
            total_top5_correct = 0
            total_ins_num = 0

            acc_dict[dataset_name] = {}
            calib_dict[dataset_name] = {}

            class_name_list = class_name_dict[dataset_name]

            zs_classifier = text_classifier_dict[dataset_name].cuda()
            # 收集当前DataSet下该Model的所有Logits和Labels
            dataset_logits_list = []
            dataset_labels_list = []

            for i, class_name in enumerate(class_name_list):
                data = image_data_dict[dataset_name][class_name].cuda()
                # 这里得到的是归一化后的点积，CLIP等模型计算Loss时通常会乘以一个logit_scale(temperature)，但这里的raw logits不包含temperature，计算ECE时可能需要后续加上temperature scaling
                pred_logits = data @ zs_classifier
                preds = torch.argmax(pred_logits, dim=1)
                labels = torch.full_like(preds, i)
                # 将数据转回CPU以免显存爆炸，并加入列表
                dataset_logits_list.append(pred_logits.cpu())
                dataset_labels_list.append(labels.cpu())

                top1_correct = (preds == labels).sum().item()

                # Calculate the class-level accuracy
                accuracy = top1_correct / len(labels)

                acc_dict[dataset_name][class_name] = accuracy

                # Calculate the top 1 acc
                total_top1_correct += top1_correct
                total_ins_num += len(labels)

                # Calculate the top 5 acc
                if int(pred_logits.shape[1]) > 5:
                    top5_preds = torch.topk(pred_logits, k=5, dim=1).indices
                    top5_correct = top5_preds.eq(labels.view(-1, 1)).sum().item()
                else:
                    top5_correct = len(labels)
                total_top5_correct += top5_correct

            # 拼接当前数据集的所有数据
            if len(dataset_logits_list) > 0:
                all_logits = torch.cat(dataset_logits_list, dim=0)
                all_labels = torch.cat(dataset_labels_list, dim=0)
                # ---------------- Start 修改部分 -----------------
                # 原始 logits 是余弦相似度，范围太小。
                # CLIP 类模型通常使用 100 作为 scaling factor。
                temperature_scale = 100.0
                scaled_logits = all_logits * temperature_scale

                # 使用缩放后的 logits 计算 softmax
                probs = torch.softmax(scaled_logits, dim=1)

                # 重新计算 ECE 和 ACE
                ece_score = calculate_ece(probs, all_labels)
                ace_score = calculate_ace(probs, all_labels)

                calib_dict[dataset_name]['ece'] = ece_score
                calib_dict[dataset_name]['ace'] = ace_score

                print(f"Dataset: {dataset_name}, ECE: {ece_score:.4f}, ACE: {ace_score:.4f}")
                # ---------------- End 修改部分 -------------------
                save_path = os.path.join(logits_save_dir, f"{model_name}__{dataset_name}.pth")
                torch.save({
                    'logits': all_logits,  # Shape: [N_samples, N_classes]
                    'labels': all_labels,  # Shape: [N_samples]
                    'acc1': total_top1_correct / total_ins_num  # 顺便存个精度方便核对
                }, save_path)

            top1_acc = total_top1_correct / total_ins_num
            top5_acc = total_top5_correct / total_ins_num

            model_performance_list.append((dataset_name, " ".join(m), m[0], m[1], top1_acc, top5_acc))

        pickle.dump(acc_dict, f)

        # --- 新增：保存 calibration 字典到 pkl ---
        with open(calib_path, 'wb') as f_calib:
            pickle.dump(calib_dict, f_calib)
        # ---------------------------------------

        csv_filename = "../LOVM/eval_table.csv"
        file_exists = os.path.isfile(csv_filename)

        # ================= save eval_table.csv =================
        # with open(csv_filename, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     if not file_exists:
        #         writer.writerow(["dataset", "model_fullname", "model","pretrained","acc1","acc5"])
        #     writer.writerows(model_performance_list)


def main():
    model_names = get_model_names()

    dataset_names, dataset_dict, class_name_dict = get_dataset_and_class_names()

    # calculate ptm's class-level gap vector
    print("==================== extract img feat and text feat ====================")
    extract_img_feat_and_text_feat(model_names, dataset_names, dataset_dict, class_name_dict)

    print("==================== calculate modality gap vector ====================")
    cal_gap_dict(model_names, dataset_names, class_name_dict)

    # calculate ptm's class-level accuracy
    print("==================== calculate class level accuracy ====================")
    cal_model_acc(model_names, dataset_names, class_name_dict)


if __name__ == "__main__":
    main()