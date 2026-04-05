from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from PIL import Image
import joblib

IMG_SIZE = 64

data = []
labels = []

classes = ['Healthy', 'Rust', 'Blight']

for label, folder in enumerate(classes):
    path = f"dataset/train/{folder}"
    for img_file in os.listdir(path):
        img = Image.open(os.path.join(path, img_file)).resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).flatten()
        data.append(img)
        labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")

print("Model trained and saved!")