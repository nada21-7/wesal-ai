# import pickle
# import os


# MODEL_PATH = "model/sign_model.pkl"

# def verify_ai_model():
#     if not os.path.exists(MODEL_PATH):
#         print(f" خطأ: لم يتم العثور على ملف الموديل في: {MODEL_PATH}")
#         return

#     try:
#         with open(MODEL_PATH, "rb") as f:
#             model = pickle.load(f)
        
#         print(" تم تحميل الموديل بنجاح")
#         print("-" * 30)
#         print(f" نوع الموديل: {type(model).__name__}")
        
#         if hasattr(model, 'classes_'):
#             classes = model.classes_
#             print(f" عدد الحروف المسجلة داخل الموديل: {len(classes)}")
#             print(f" الحروف الموجودة فعلياً: {list(classes)}")
            
#             if len(classes) == 0:
#                 print(" تحذير: الموديل فارغ ولا يحتوي على أي حروف للترجمة")
#         else:
#             print(" الموديل لا يحتوي على خاصية 'classes_'، قد يحتاج لإعادة تدريب.")

#     except Exception as e:
#         print(f" حدث خطأ أثناء فحص الموديل: {e}")

# if __name__ == "__main__":

#     verify_ai_model()

import pickle
import os


MODEL_PATH = "model/sign_model.pkl"

def verify_ai_model():
    if not os.path.exists(MODEL_PATH):
        print(f" خطأ: لم يتم العثور على ملف الموديل في: {MODEL_PATH}")
        return

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        print(" تم تحميل الموديل بنجاح")
        print("-" * 30)
        print(f" نوع الموديل: {type(model).__name__}")
        
        if hasattr(model, 'classes_'):
            classes = model.classes_
            print(f" عدد الحروف المسجلة داخل الموديل: {len(classes)}")
            print(f" الحروف الموجودة فعلياً: {list(classes)}")
            
            
            if len(classes) == 0:
                print(" تحذير: الموديل فارغ ولا يحتوي على أي حروف للترجمة")
        else:
            print(" الموديل لا يحتوي على خاصية 'classes_'، قد يحتاج لإعادة تدريب.")

    except Exception as e:
        print(f" حدث خطأ أثناء فحص الموديل: {e}")

if __name__ == "__main__":
    verify_ai_model()
