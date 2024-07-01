from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터를 훈련셋과 테스트셋으로 분할 (70% 훈련, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForest 분류기 모델 생성 및 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델의 정확도 평가
print(f"Model accuracy: {model.score(X_test, y_test)}")

# 훈련된 모델을 파일로 저장
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")