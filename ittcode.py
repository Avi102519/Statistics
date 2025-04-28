import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Create dummy image data (small colored squares)
X = []
y = []

for i in range(100):
    img = np.random.randint(0, 256, (32, 32, 3))  # Random 32x32 color image
    label = "Cat" if np.mean(img) > 127 else "Dog"  # Simple rule: bright = cat, dark = dog
    X.append(img.flatten())  # Flatten image into 1D array
    y.append(label)

X = np.array(X)
y = np.array(y)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a simple Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Save model as .pkl
with open("image_to_text_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as image_to_text_model.pkl")
