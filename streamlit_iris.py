import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io

# 1️⃣ 웹에서 IRIS 데이터 로드
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# 웹에서 CSV 데이터를 불러와 DataFrame으로 변환
response = requests.get(url)
iris_data = response.text 

# CSV 형태로 변환 후 DataFrame 생성
df = pd.read_csv(io.StringIO(iris_data), header=None, names=columns)

# 2️⃣ Streamlit 웹페이지 설정
st.title("🌸 IRIS 데이터 시각화")
st.write("Iris 데이터셋을 웹에서 로드하여 Pandas DataFrame으로 변환한 뒤, Seaborn을 이용해 시각화합니다.")

# 3️⃣ 데이터프레임 출력
st.subheader("📋 IRIS 데이터 테이블")
st.dataframe(df)

# 4️⃣ Seaborn 그래프 그리기
st.subheader("📊 Seaborn Pairplot")
fig = sns.pairplot(df, hue="species", palette="husl")
st.pyplot(fig)

# 5️⃣ 개별 그래프 (예: Sepal Width vs Sepal Length)
st.subheader("📈 Sepal Width vs Sepal Length")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=ax, palette="husl")
st.pyplot(fig)
