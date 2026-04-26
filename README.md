# 🚀 Training

This repository provides training scripts for two multimodal sentiment analysis datasets: **MVSA-Single** and **MVSA-Multiple**.

---

## 📦 Dataset Preparation

Before training, please make sure that the datasets and pretrained BERT model are placed in the correct directories.

### Expected Directory Paths

```text
/root/autodl-tmp/cvpr/dataset/MVSA_Single/r-MVSA-S
/root/autodl-tmp/cvpr/dataset/MVSA/r-MVSA
/root/autodl-tmp/model/bert-base-uncased
```

---

## 🧪 Training on MVSA-Single

Run the following command to train the model on the **MVSA-Single** dataset.

```bash
python main.py \
  --data_dir /root/autodl-tmp/cvpr/dataset/MVSA_Single/r-MVSA-S \
  --train_data_dir /root/autodl-tmp/cvpr/dataset/MVSA_Single/r-MVSA-S \
  --test_data_dir /root/autodl-tmp/cvpr/dataset/MVSA_Single/r-MVSA-S \
  --bert_model_path /root/autodl-tmp/model/bert-base-uncased \
  --gpu 0 \
  --save_dir "./mvsa-s/600/" \
  --lr 1e-4 \
  --num_labels 600 \
  --batch_size 2 \
  --num_train_iter 256 \
  --threshold 0.95
```

---

## 🧪 Training on MVSA-Multiple

Run the following command to train the model on the **MVSA-Multiple** dataset.

```bash
python main.py \
  --data_dir /root/autodl-tmp/cvpr/dataset/MVSA/r-MVSA \
  --train_data_dir /root/autodl-tmp/cvpr/dataset/MVSA/r-MVSA \
  --test_data_dir /root/autodl-tmp/cvpr/dataset/MVSA/r-MVSA \
  --bert_model_path /root/autodl-tmp/model/bert-base-uncased \
  --gpu 0 \
  --save_dir "./mvsa-m/1500/" \
  --lr 1e-4 \
  --num_labels 1500 \
  --batch_size 2 \
  --num_train_iter 256 \
  --threshold 0.95
```

---

## ⚙️ Main Configuration

| Argument | Setting |
| --- | --- |
| `--bert_model_path` | `bert-base-uncased` |
| `--gpu` | `RTX 4080 Super` |
| `--lr` | `1e-4` |
| `--batch_size` | `2` |
| `--num_train_iter` | `256` |
| `--threshold` | `0.95` |
