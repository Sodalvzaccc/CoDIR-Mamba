## 🚀 Training

We provide running scripts for two widely used multimodal sentiment analysis datasets: **MVSA-Single** and **MVSA-Multiple**.

### 📌 Train on MVSA-Single

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
