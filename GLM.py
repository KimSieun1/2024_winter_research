from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#############################
# DataFrame(테이블 형태의 데이터를 다루기 위한 2차원 데이터 구조)을 만들어 ApEn 저장

df = pd.DataFrame(columns = ['Mean','SD'])

file_path = ['Normal_Mean_SD.txt', 'Depression_Mean_SD.txt', 
'PD_Mean_SD.txt'] #file_path에 파일 명으로 된 문자열 리스트를 저장 

for i in range(len(file_path)):  
    with open(file_path[i], 'r') as file : #file_path안의 파일명을 전달. 그 파일을 열어서 읽음
        for line in file:
            data = line.strip().split()
            if len(data) == 2 and all(val.replace('.', '', 1).isdigit() for val in data):  #현재 읽은  라인이 2개의 열로 구성되었는지/모든 열이 숫자인지 확인(소수점을 제거하고 봄)
                new_df = pd.DataFrame({'Mean': [float(data[0])], 'SD': [float(data[1])]})  #두 조건이 모두 충족되면, 현재 라인의 데이터를 새로운 데이터프레임(new_df)으로 변환
                if not df.empty:    # 기존 데이터 프레임이 비어있지 않다면
                    df = pd.concat([df, new_df], ignore_index=True)   # 기존 데이터프레임과 새로운 데이터프레임을 연결 / 인덱스를 무시하고 새로운 인덱스를 할당하는 옵션
                else:
                    df = new_df.copy()  #비어있다면, 새로운 데이터프레임을 복사하여 df에 할당

df['label'] = ['N','N','N','N','N','N','N','N','N','N','D','D','D','D','D','D','D','D','D','D','P','P','P','P','P','P','P','P','P','P'] #라벨 추가
print(df) 

#######################################
# 저장한 데이터프레임의 정보 시각화

sns.pairplot(df, hue='label', vars=['Mean', 'SD'])
plt.show()

#######################################
#훈련 데이터와 테스트 데이터 분리

train_input, test_input, train_target, test_target = train_test_split(
    df[['Mean', 'SD']], df['label'], random_state=42)  #random_state가 같으면 같은 난수 형성 / train :22개, test : 8개

#표준점수로 데이터 스케일링(정확도를 높이기 위한 데이터 전처리, 각 feature의 범위 맞추기)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)
#print(test_input,test_target,train_input, train_target)

########################################
#GLM
lr = LogisticRegression(max_iter=50)  # 학습시키는 알고리즘 실사화 #LogisticRegression 객체를 생성 # 최적화를 위한 최대반복횟수 =50 

#로지스틱 회귀 학습
lr.fit(train_scaled, train_target)  #실사화한 알고리즘으로, 훈련 데이터(train_scaled)와 해당 레이블(train_target)을 사용하여 모델을 학습, 결정 경계 탐색 

#테스트 데이터 예측
pred = lr.predict(test_scaled) #테스트 데이터(test_scaled)의 처음 5개 샘플을 사용하여 label 예측을 수행
print(test_scaled) #테스트 실행한 데이터 출력
print(test_target) #테스트 실행한 데이터의 답(label) 출력
print(pred) #예측한 label 출력

###########################################
# Confusion matrix를 이용한 성능 평가

cm = confusion_matrix(test_target, pred)
accuracy = accuracy_score(test_target, pred)
precision = precision_score(test_target, pred, average='weighted')
recall = recall_score(test_target, pred, average='weighted')
f1 = f1_score(test_target, pred, average='weighted')

# 결과 출력
print("GLM Score :")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)