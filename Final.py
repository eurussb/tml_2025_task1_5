import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import time

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]
#extract loss and label from each image
def extract_features(dataset, model):
    features, labels = [], []
    for i in range(len(dataset)):
        _, img, label, membership = dataset[i]
        img = img.unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            logits = output[0]
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label])).item()
            feature_vector = [loss, label]
            features.append(feature_vector)
            labels.append(membership)
    return np.array(features), np.array(labels)

#helper functions
def compute_probs(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
    return probs

def estimate_pr(model, inputs, labels):
    probs = compute_probs(model, inputs)
    return probs[range(len(labels)), labels]

def estimate_pr_ensemble(models, inputs, labels):
    with torch.no_grad():
        probs_list = [torch.softmax(model(inputs), dim=1) for model in models]
        avg_probs = torch.stack(probs_list).mean(dim=0)
    return avg_probs[range(len(labels)), labels]


batch_size = 128
normalize_only = transforms.Normalize(mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889])
#load public data set
pub_data = torch.load("/home/adugast/tml_2025_task1/pub.pt", map_location=torch.device('cpu'))
pub_dataset = MembershipDataset(transform=normalize_only)
for entry in pub_data:
    id_, img, label, member = entry
    pub_dataset.ids.append(id_)
    pub_dataset.imgs.append(img)
    pub_dataset.labels.append(label)
    pub_dataset.membership.append(member)

pub_loader = DataLoader(pub_dataset, batch_size=batch_size, shuffle=False)
# use nonmember images as z
nonmember_pool = [(id_, img, label) for id_, img, label, member in zip(
    pub_dataset.ids, pub_dataset.imgs, pub_dataset.labels, pub_dataset.membership
) if member == 0]

target_model = resnet18(weights=False)
target_model.fc = nn.Linear(512, 44)
target_model.load_state_dict(torch.load("/home/adugast/tml_2025_task1/01_MIA.pt", map_location=torch.device('cpu')))
target_model.eval()

reference_model_paths = [
    "model1_resnet18_sgd_trained.pth",
    "model2_resnet18_adamw_trained.pth",
]

reference_models = []
for path in reference_model_paths:
    model = resnet18(weights=False)
    model.fc = nn.Linear(512, 44)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    reference_models.append(model)

priv_data = torch.load("/home/adugast/tml_2025_task1/priv_out.pt", map_location=torch.device('cpu'))
priv_dataset_with_membership = MembershipDataset(transform=normalize_only)
for entry in priv_data:
    id_, img, label, _ = entry
    priv_dataset_with_membership.ids.append(id_)
    priv_dataset_with_membership.imgs.append(img)
    priv_dataset_with_membership.labels.append(label)
    priv_dataset_with_membership.membership.append(0)

priv_loader = DataLoader(priv_dataset_with_membership, batch_size=batch_size, shuffle=False)

gamma = 32
a= 0.4
   
rmia_scores = []
true_membership = []        
#compute RMIA for pub, to train the classifier
for batch_ids, batch_imgs, batch_labels, batch_memberships in pub_loader:
    current_ids = set(batch_ids.numpy())
    filtered_pool = [(id_, img, label) for id_, img, label in nonmember_pool if id_ not in current_ids]
    sampled_Z = random.sample(filtered_pool, 250)
    Z_imgs = torch.stack([normalize_only(img) for _, img, _ in sampled_Z])
    Z_labels = torch.tensor([label for _, _, label in sampled_Z])
    pr_target_Z = estimate_pr(target_model, Z_imgs, Z_labels)
    pr_out_Z = estimate_pr_ensemble(reference_models, Z_imgs, Z_labels)
    Z_ratio = pr_target_Z / (pr_out_Z)
    Z_ratio_np = Z_ratio.numpy()
    pr_target = estimate_pr(target_model, batch_imgs, batch_labels).numpy()
    pr_out = estimate_pr_ensemble(reference_models, batch_imgs, batch_labels)
    pr_combined = 0.5 * ((1 + a) * pr_out + (1 - a))
    ratio_x = pr_target / (pr_combined)
    comparison = (ratio_x[:, None] / Z_ratio_np[None, :]) > gamma
    score_mia_batch = np.mean(comparison.numpy(), axis=1)
    rmia_scores.extend(score_mia_batch.tolist())
    true_membership.extend(batch_memberships.numpy().tolist())

pub_features, _ = extract_features(pub_dataset, target_model)
X_train = np.column_stack([np.array(rmia_scores), pub_features])
y_train = np.array(true_membership)
#train the classifer
base_attack_model = LogisticRegression(solver='lbfgs', max_iter=5000)
attack_model = CalibratedClassifierCV(estimator=base_attack_model, method='sigmoid', cv=10)
attack_model.fit(X_train, y_train)

priv_scores = []

Z_imgs = torch.stack([normalize_only(img) for _, img, _ in nonmember_pool])
Z_labels = torch.tensor([label for _, _, label in nonmember_pool])

pr_target_Z = estimate_pr(target_model, Z_imgs, Z_labels)
pr_out_Z = estimate_pr_ensemble(reference_models, Z_imgs, Z_labels)
Z_ratio = pr_target_Z / (pr_out_Z)
Z_ratio_np = Z_ratio.numpy()
#compute RMIA for priv_out and use the classifier to infer membership
for batch_ids, batch_imgs, batch_labels, _ in priv_loader:
    pr_target = estimate_pr(target_model, batch_imgs, batch_labels).numpy()
    pr_out = estimate_pr_ensemble(reference_models, batch_imgs, batch_labels).numpy()
    pr_combined = 0.5 * ((1 + a) * pr_out + (1 - a))
    ratio_x = pr_target / pr_combined
    score_mia_batch = np.mean((ratio_x[:, None] / Z_ratio_np[None, :]) > gamma, axis=1)
    priv_scores.extend(score_mia_batch.tolist())
priv_features, _ = extract_features(priv_dataset_with_membership, target_model)
X_priv = np.column_stack([np.array(priv_scores), priv_features])
membership_scores = attack_model.predict_proba(X_priv)[:, 1]
df = pd.DataFrame({
    "ids": priv_dataset_with_membership.ids,
    "score": membership_scores
})
csv_filename = f"test_a_{a:.1f}.csv"
df.to_csv(csv_filename, index=None)
response = requests.post("http://34.122.51.94:9090/mia", files={"file": open(csv_filename, "rb")}, headers={"token": "85928950"})
result = response.json()
with open("response_log.txt", "a") as log_file:
    log_file.write(f"Config a={a:.1f}, gamma={gamma}, batch_size={batch_size}\n")
    log_file.write(f"Response: {result}\n\n")
       

    