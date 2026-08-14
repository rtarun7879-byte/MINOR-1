import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv(r"D:\veltech summerintenship\Veltech-Summer-Internship\CropSense\crop production.csv")

X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

le = LabelEncoder()
y_enc = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_enc)

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model trained and saved successfully!")