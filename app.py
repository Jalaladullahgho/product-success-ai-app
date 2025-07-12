import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# البيانات
data = {
    'price': [10, 25, 50, 15, 40, 70, 20, 60],
    'category_score': [8.5, 6.0, 9.0, 5.0, 7.5, 4.5, 6.8, 9.5],
    'prelaunch_interest': [100, 60, 150, 40, 80, 30, 55, 160],
    'reviews_expected': [4.5, 3.0, 4.8, 2.5, 4.0, 2.0, 3.2, 4.9],
    'product_success': [1, 0, 1, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# التدريب
X = df.drop('product_success', axis=1)
y = df['product_success']
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit App
st.title("🔍 هل سينجح المنتج؟")
st.write("أدخل خصائص المنتج لمعرفة التوقع:")

price = st.slider("سعر المنتج", 5, 100, 30)
category_score = st.slider("درجة الفئة (شعبية السوق)", 0.0, 10.0, 7.0)
interest = st.slider("عدد المهتمين قبل الإطلاق", 0, 200, 100)
reviews = st.slider("متوسط التقييم المتوقع", 1.0, 5.0, 4.0)

# التنبؤ
sample = np.array([[price, category_score, interest, reviews]])
prediction = model.predict(sample)

if st.button("توقع النجاح"):
    result = "✅ سينجح!" if prediction[0] == 1 else "❌ سيفشل"
    st.subheader(f"النتيجة: {result}")
