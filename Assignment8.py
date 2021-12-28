#neural network함수 불러오기
from sklearn.neural_network import MLPClassifier
#데이터셋과 시각화용 라이브러리
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#파일 불러오기 위해서 (모델 형성) 필요한 라이브러
import pandas as pd

#파일 불러오기
df = pd.read_csv("student_health_2.csv", encoding = "cp949")

#간단한 데이터셋 생성
X = df[["키", "몸무게"]]
y = df[["학년"]]

#데이터 분리 => 필요없음(데이터셋을 만들 때 분리해서 만들었음.)
#X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, random_state=1)


#neural network 모델 적합
clf = MLPClassifier(solver = "lbfgs", alpha = 1e-5, hidden_layer_sizes = (5,2), random_state = 1)
clf.fit(X,y)

#예측
my_body = [[150, 45]]
my_predict = clf.predict(my_body)
print("예측 값은")
print(my_predict)

# 5. weights 확인
print("weights는")
print(clf.coefs_)

#5. MLP 점수 산정 = 정확률 계산
print("정확률은")
print(clf.score(X, y))

