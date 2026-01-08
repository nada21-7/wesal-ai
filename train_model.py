import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

dataset_path = "dataset"
X = []
y = []

print(" فحص الملفات وجمع البيانات...")

for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        file_path = os.path.join(dataset_path, file)
        count_in_file = 0 
        
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            try:
                header = next(reader) 
            except StopIteration:
                print(f" الملف {file} فارغ تماماً!")
                continue
            
            for row in reader:
                try:
                    numeric_row = [float(val) for val in row if val.replace('.','',1).replace('-','',1).isdigit()]
                    
                    if len(numeric_row) >= 42:
                        X.append(numeric_row[:42])
                        y.append(label)
                        count_in_file += 1
                except:
                    continue
        
        if count_in_file > 0:
            print(f" تم تحميل {count_in_file} سطر من ملف: {file}")
        else:
            print(f" الملف {file} لا يحتوي على بيانات رقمية صالحة (قد يكون به عناوين فقط).")

if not X:
    print(" خطأ: لا توجد أي بيانات صالحة للتدريب في المجلد!")
else:
    X = np.array(X)
    y = np.array(y)

    print("-" * 30)
    print(f" إجمالي عينات التدريب: {len(X)}")
    print(f" عدد الحروف : {len(np.unique(y))}")
    print(f" قائمة الحروف المبرمجة: {list(np.unique(y))}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    if not os.path.exists("model"): os.makedirs("model")
    with open("model/sign_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("الموديل جاهز الآن للعمل!")
