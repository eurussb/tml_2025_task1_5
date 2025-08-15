import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd

# own imports
from torchvision import transforms
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch.nn.functional as F

#### LOADING THE MODEL

from torchvision.models import resnet18

### Add this as a transofrmation to pre-process the images
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]
transform = transforms.Normalize(mean=mean, std=std)

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA.pt", map_location="cpu")

model.load_state_dict(ckpt)
model.eval()

#### DATASETS

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]
    
### Feature extraction

def extract_features(dataset, model):
    features, labels = [], []
    for i in range(len(dataset)):
        _, img, label, membership = dataset[i]
        img = img.unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            logits = output[0]
            probs = F.softmax(logits, dim=0)

            confidence = probs[label].item()
            entropy = -torch.sum(probs * torch.log(probs)).item()            
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label])).item()
            logits_list = logits.tolist()

            feature_vector = [confidence,entropy, loss, *logits_list]
            features.append(feature_vector)
            labels.append(membership)
    return np.array(features), np.array(labels)

### Load pub data and train the attack model

pub_data = torch.load("./pub.pt", weights_only=False)
pub_data.transform = transform

X_pub, y_pub = extract_features(pub_data, model)
imputer = SimpleImputer(strategy="mean")
X_pub_imputed = imputer.fit_transform(X_pub)


attack_model = xgb.XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.03,
    max_depth=3,
    n_estimators=80,
    subsample=0.8,
    eval_metric='logloss'
)
#attack_model = RandomForestClassifier(
#    n_estimators=200,
#    max_depth=10,
#    min_samples_split=5,
#    min_samples_leaf=1,
#    max_features='sqrt',
#    class_weight='balanced',
#    random_state=42
#)

attack_model.fit(X_pub_imputed, y_pub)

### Load the priv dataset and compute the confidence score for each image

data: MembershipDataset = torch.load("./priv_out.pt", weights_only=False)
data.transform = transform

X_priv, _ = extract_features(data, model)
X_priv_imputed = imputer.transform(X_priv)

membership_scores = attack_model.predict_proba(X_priv_imputed)[:, 1]

df = pd.DataFrame({
    "ids": data.ids,
    "score": membership_scores
})
df.to_csv("test.csv", index=None)
response = requests.post("http://34.122.51.94:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "85928950"})
print(response.json())
