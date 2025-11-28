
import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import class_weight

# --- 1. Kaggle Data Setup ---
def setup_kaggle_and_download(dataset_handle, download_path="./data"):
    """
    يجهز بيئة Kaggle ويحمل البيانات تلقائياً.
    يتطلب وجود ملف kaggle.json في المجلد الرئيسي.
    """
    print("📥 Setting up Kaggle environment...")
    
    # التأكد من وجود ملف التوكين
    if not os.path.exists("kaggle.json"):
        raise FileNotFoundError("❌ Please upload 'kaggle.json' to the root directory!")

    # إعداد المجلدات
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    # التحميل
    if not os.path.exists(f"{download_path}/chest_xray"):
        print(f"⬇️ Downloading {dataset_handle}...")
        import kaggle
        kaggle.api.dataset_download_files(dataset_handle, path=download_path, unzip=True)
        print("✅ Download & Unzip Complete.")
    else:
        print("✅ Data already exists.")

    return os.path.join(download_path, "chest_xray")

# --- 2. Data Partitioning (Dirichlet) ---
def get_data_partitions(data_dir, alpha, num_clients, img_size=128):
    print(f"📊 Partitioning Data (Alpha={alpha})...")
    
    # تحميل البيانات كـ Arrays للتحكم في التقسيم
    train_dir = os.path.join(data_dir, "train")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(img_size, img_size), batch_size=None, 
        shuffle=True, labels='inferred', label_mode='int', color_mode='rgb'
    )
    
    images, labels = [], []
    for img, lbl in train_ds:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    
    x_train = np.array(images) / 255.0
    y_train = np.array(labels)

    # خوارزمية ديريخليه
    min_size = 0
    N = y_train.shape[0]
    client_datasets = []
    
    while min_size < 10: # ضمان عدم وجود عميل فارغ
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(2): # 2 classes
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

    for i in range(num_clients):
        client_datasets.append((x_train[idx_batch[i]], y_train[idx_batch[i]]))
    
    # Test Set
    test_dir = os.path.join(data_dir, "test")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(img_size, img_size), batch_size=32, shuffle=False, color_mode='rgb'
    )
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))
    
    return client_datasets, test_ds

# --- 3. Helper Functions ---
def calculate_class_weights(y_train):
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    return {k: v for k, v in zip(np.unique(y_train), weights)}

# --- 4. Saving & Plotting Results (The Important Part) ---
def save_experiment_results(history, client_datasets, config):
    """
    يقوم بإنشاء مجلد للتجربة وحفظ المخططات والنتائج بداخله.
    """
    exp_dir = f"./results/{config['experiment_name']}"
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)
    
    print(f"💾 Saving results to: {exp_dir}")

    # 1. حفظ مخطط توزيع البيانات
    plt.figure(figsize=(10, 6))
    client_ids = range(len(client_datasets))
    normal_counts = [np.sum(y == 0) for _, y in client_datasets]
    pneumonia_counts = [np.sum(y == 1) for _, y in client_datasets]
    
    plt.bar(client_ids, normal_counts, label='Normal')
    plt.bar(client_ids, pneumonia_counts, bottom=normal_counts, label='Pneumonia')
    plt.title(f"Data Distribution (Alpha={config['data']['alpha']})")
    plt.xlabel("Client ID")
    plt.ylabel("Samples")
    plt.legend()
    plt.savefig(f"{exp_dir}/data_distribution.png")
    plt.close()

    # 2. حفظ مخططات الأداء (Loss & Accuracy & AUC)
    rounds = [i for i in range(1, len(history.losses_distributed) + 1)]
    
    # Loss
    plt.figure()
    # ملاحظة: Flower history structure قد يختلف قليلاً حسب النسخة
    # هنا نفترض استخدام evaluate_fn المركزي الذي يعيد losses_centralized
    loss_values = [x[1] for x in history.losses_centralized]
    plt.plot(rounds, loss_values, marker='o', label='Centralized Loss')
    plt.title(f"Loss Progression ({config['server']['strategy']})")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{exp_dir}/loss_curve.png")
    plt.close()

    # Metrics (Accuracy & AUC)
    if 'accuracy' in history.metrics_centralized:
        acc_values = [x[1] for x in history.metrics_centralized['accuracy']]
        plt.figure()
        plt.plot(rounds, acc_values, marker='s', color='green', label='Accuracy')
        plt.title("Accuracy Progression")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig(f"{exp_dir}/accuracy_curve.png")
        plt.close()

    # 3. حفظ ملف نصي بالنتائج النهائية والإعدادات
    with open(f"{exp_dir}/summary.txt", "w") as f:
        f.write(f"Experiment: {config['experiment_name']}\n")
        f.write("="*30 + "\n")
        f.write(f"Strategy: {config['server']['strategy']}\n")
        f.write(f"Alpha: {config['data']['alpha']}\n")
        f.write(f"Client Epochs: {config['client']['local_epochs']}\n")
        f.write(f"Optimizer: {config['client']['optimizer']} (LR={config['client']['learning_rate']})\n")
        f.write("-" * 20 + "\n")
        f.write("Final Results:\n")
        f.write(f"Final Loss: {loss_values[-1]:.4f}\n")
        if 'accuracy' in history.metrics_centralized:
            f.write(f"Final Accuracy: {acc_values[-1]:.4f}\n")
            
    # 4. نسخ ملف الكونفيج للأرشيف
    shutil.copy("config.yaml", f"{exp_dir}/config_backup.yaml")
    
    print("✅ All results saved successfully.")
