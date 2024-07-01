from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 생성 및 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델을 파일로 저장
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# 저장된 모델 로드
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# 로드된 모델 사용
predictions = loaded_model.predict(X_test)