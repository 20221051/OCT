import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models, datasets
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Validation loss did not improve for {} epochs. Stopping...".format(self.patience))
                return True
        return False

class BaseTransform() :
    def __init__(self) :
        self.base_transform = transforms.Compose([
            lambda x: x.convert("RGB"),
            transforms.Resize((256, 256)), # 짧은 변의 길이 기준으로 resize
            transforms.ToTensor(), # 토치 텐서로 변환
            transforms.ConvertImageDtype(torch.float),
        ])

    def __call__(self, img) :
        img = img.crop((0, 0, img.width, img.height - (img.height // 4)))

        return self.base_transform(img)

class CustomDataset(Dataset):
    def __init__(self, image_files, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        for i in range(len(root_dir)):
            img_name = os.path.join(self.root_dir, self.image_files[idx])
            image = Image.open(img_name)
            try:
                label = float(img_name.split('VA ')[-1].split('.png')[0])  # 파일 이름에서 레이블 추출
            except ValueError:
                label = float(0.01)
            if label == 7.0:
                label =4
            elif label >=0.8:
                label = 4
            elif label >=0.5:
                label = 3
            elif label >=0.3:
                label = 2
            elif label >=0.1:
                label = 1
            else:
                lable = 0
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor([label], device='cuda:0').type(torch.LongTensor)

class ConcatDataset(Dataset):
    def __init__(self, grouped_image_files, root_dir, transform=None):
        self.grouped_image_files = grouped_image_files  # 이미지 파일 그룹
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.grouped_image_files)

    def __getitem__(self, idx):
        image_group = self.grouped_image_files[idx]
        # 이미지 그룹에서 이미지를 로드하고 합칩니다.
        images = [Image.open(os.path.join(self.root_dir, img_name)) for img_name in image_group]
        # 예를 들어, 이미지를 가로로 합칩니다.
        concat_image = Image.new('RGB', (images[0].width * len(images), images[0].height))
        for i, img in enumerate(images):
            concat_image.paste(img, (i * img.width, 0))

        if self.transform:
            concat_image = self.transform(concat_image)

        label = self.extract_label(image_group[0])
        return concat_image, label

    def extract_label(self, img_name):
        label = float(img_name.split('VA ')[-1].split('.png')[0])
        return label

resize = 224

transform = BaseTransform()

# 데이터셋 및 데이터로더 생성
dataset = CustomDataset( image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')] ,root_dir=root_dir, transform=transform)

len(os.listdir()) # 전체 이미지 개수

image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

len(list(set([img_name[0:7] for img_name in dataset.image_files]))) # 환자 ID 당 이미지 개수

label_info = list((img_name.split('VA ')[-1].split('.png')[0]) for img_name in dataset.image_files)

def convert_to_float_or_default(value):
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        float_value = 0.01
    return float_value
converted_list = [convert_to_float_or_default(element) for element in label_info]
converted_list = np.array(converted_list)
np.where(converted_list>2)[0]
converted_list[np.where(converted_list>2)[0]]=0.7

import matplotlib.pyplot as plt
import numpy as np

# 예시 데이터 생성
data = np.random.randn(1000)

# 히스토그램 그리기
plt.hist(converted_list, bins=100, color='blue', alpha=0.7)
plt.title('VA')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 그래프 보여주기
plt.show()

# 환자 ID 목록 추출
patient_ids = list(set([img_name[0:7] for img_name in dataset.image_files]))

# 환자 ID를 기준으로 train, val, test set으로 나누기
train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

train_ids_list = list(set([img_name for img_name in dataset.image_files if img_name[0:7] in train_ids]))
val_ids_list = list(set([img_name for img_name in dataset.image_files if img_name[0:7] in val_ids]))
test_ids_list = list(set([img_name for img_name in dataset.image_files if img_name[0:7] in test_ids]))

train_dataset = CustomDataset(image_files=train_ids_list, root_dir=root_dir, transform=transform)


val_dataset = CustomDataset(image_files=val_ids_list, root_dir=root_dir, transform=transform)


test_dataset = CustomDataset(image_files=test_ids_list, root_dir=root_dir, transform=transform)


# 데이터로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

model = models.vgg16(pretrained=True)
num_features = model.classifier[6].in_features
model.classifier[6] =nn.Sequential(
    nn.Linear(num_features, 5),
    nn.Softmax()
)   # 다중 분류를 위한 출력 뉴런 5개로 변경

# 손실 함수와 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Early Stopping 설정
early_stopping = EarlyStopping(patience=20, verbose=True)

# 학습 코드
num_epochs = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_list = []
val_loss_list = []
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze(-1) ) # CrossEntropy 손실 사용
        loss.backward()
        optimizer.step()
        print("Train loss: ", loss.item())
        running_loss += loss.item()

    # Validation Loss 계산
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels.reshape(-1)).item()
            print("Val loss: ", criterion(outputs, labels.reshape(-1)).item())
    # Validation Loss 기준으로 Early Stopping 수행
    if early_stopping(val_loss):
        print("Early stopping.")
        break
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_dataloader)}, Val Loss: {val_loss/len(val_dataloader)}')
    train_loss_list.append(running_loss / len(train_dataloader))
    val_loss_list.append(val_loss / len(val_dataloader))

# loss 곡선
plt.plot(range(1, len(train_loss_list)+1), train_loss_list, label='Training Loss')
plt.plot(range(1, len(val_loss_list)+1), val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

model.load_state_dict(torch.load('best_model.pth'))
model.to(device)

# AUC
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc

# ... (이전 코드와 필요한 라이브러리 및 클래스를 import)

# 모델 평가 모드로 설정
model.eval()

correct_predictions = 0
total_samples = 0
all_predictions = []
all_labels = []
predict_proba = []

# 예측값과 실제값 비교
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).cpu()
        predict_proba.extend(outputs.numpy())
        predictions = np.argmax(outputs,axis=1)

#         predictions = (outputs > 0.5).float()  # 이진 분류의 경우, 0.5를 기준으로 이상인지 이하인지 결정
        for i in range(len(predictions)):
            if predictions[i] == labels.reshape(-1)[i]:
                correct_predictions += 1
            total_samples += 1

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.reshape(-1).cpu().numpy())

accuracy = correct_predictions / total_samples

# F1 Score 계산
f1_micro = f1_score(all_labels, all_predictions,average = 'micro').astype(int)
f1_macro = f1_score(all_labels, all_predictions,average = 'macro').astype(int)
f1_weight = f1_score(all_labels, all_predictions,average = 'weighted').astype(int)

# AUC 및 ROC Curve 계산
lb = LabelBinarizer()
true_labels_bin = lb.fit_transform(all_labels)
roc_auc_scores = []
fpr = dict()
tpr = dict()
roc_aucc = dict()
for i in range(len(lb.classes_)):
    roc_auc = roc_auc_score(true_labels_bin[:, i], [p[i] for p in predict_proba])
    roc_auc_scores.append(roc_auc)
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], [p[i] for p in predict_proba])
    roc_aucc[i] = auc(fpr[i],tpr[i])

roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)

n_classes = len(set(all_labels))

# 그래프 그리기
plt.figure(figsize=(6, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_aucc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.show()

# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()

print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'F1 Score: {f1_micro:.2f},{f1_macro:.2f},{f1_weight:.2f}')
print(f'AUC: {roc_auc:.2f}')

# Grad cam

def grad_cam(model, input_tensor, target_class):
    model.eval()

    # 모델의 특정 레이어 가져오기 (예시로 마지막 컨볼루션 레이어를 사용합니다)
    target_layer = model.features[-1]

    # 입력 이미지에 대한 그래디언트 계산
    input_tensor.requires_grad_()
    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()

    # 해당 레이어에서의 그래디언트 얻기
    grads = input_tensor.grad
    pooled_grads = torch.mean(grads, dim=(2, 3), keepdim=True)

    # 해당 레이어의 출력(feature map) 얻기
    target = target_layer(input_tensor)
    target = F.relu(target)

    # 각 채널의 중요도 계산
    weights = F.adaptive_avg_pool2d(pooled_grads, (target.shape[2], target.shape[3]))
    cam = torch.sum(weights * target, dim=1, keepdim=True)

    # Normalize
    cam = F.relu(cam)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)

    return cam.squeeze()
def resize_gradcam(cam, target_size):
    return F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze()

from PIL import Image
from torch.nn import functional as F
# ... (이전 코드와 필요한 라이브러리 및 클래스를 import)

# 모델 평가 모드로 설정
model.eval()

# 테스트 데이터셋으로부터 이미지와 레이블 가져오기
predict_proba = []
for i, (inputs, labels) in enumerate(test_dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs).cpu()
    predict_proba.extend(outputs.detach().numpy())
    predictions = np.argmax(outputs.detach().numpy(),axis=1)
    # 이미지와 레이블 시각화
    for j in range(len(predictions)):
        image = inputs[j].cpu().numpy().transpose((1, 2, 0))
        label = labels[j].item()
        prediction = predictions[j].item()

        # Grad-CAM 계산
        target_class = int(prediction)
        cam = grad_cam(model, inputs[j].unsqueeze(0), target_class)
        cam = resize_gradcam(cam, inputs[j].shape[1:])

        label_info = ['VA < 0.1','0.1 <= VA < 0.3', '0.3<= VA < 0.5', '0.5 <= VA < 0.8', 'VA >= 0.8']
        colors = ['orange' for _ in label_info]
        special_label = label_info[label]
        special_color = 'orange'
        if special_label in label_info:
            special_index = label_info.index(special_label)
            colors[special_index] = special_color

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Sample {i * test_dataloader.batch_size + j + 1}\nActual Label: {label}")
        # Grad-CAM 시각화
        plt.subplot(1, 3, 2)
        plt.imshow(cam.cpu().detach().numpy(), cmap='jet', alpha=1)
        plt.subplot(1, 3, 2)
        plt.imshow(image, alpha=0.2)
        plt.title(f"Grad-CAM for Class {target_class}")

        plt.subplot(1, 3, 3)
        plt.xticks(rotation=45,fontsize=8)
        plt.bar(label_info,outputs.detach()[j] , color=colors)
        plt.title(f"Actual Label: {label}, Predicted Label: {np.argmax(outputs.detach()[j])}")
        plt.ylim(0, 1)
        plt.show()

        if i * test_dataloader.batch_size + j == 100:  # 예제로 10개까지만 보여줍니다. 필요에 따라 수정 가능
            break

    if i * test_dataloader.batch_size + j == 100:
        break
