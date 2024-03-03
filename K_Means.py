from sklearn.metrics import confusion_matrix    # confusion_matrix(예측 오류값과 어떤 유형의 오류인지 나타내는 지표)라이브러리
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#####################################
# DataFrame(테이블 형태의 데이터를 다루기 위한 2차원 데이터 구조)을 만들어 ApEn 저장

df = pd.DataFrame(columns = ['x','y'])

file_path = ['Normal_Mean_SD.txt', 'Depression_Mean_SD.txt', 'PD_Mean_SD.txt'] #file_path에 파일 명으로 된 문자열 리스트를 저장 

for i in range(len(file_path)):  
    with open(file_path[i], 'r') as file : #file_path안의 파일명을 전달. 그 파일을 열어서 읽음
        for line in file:
            data = line.strip().split()
            if len(data) == 2 and all(val.replace('.', '', 1).isdigit() for val in data):  #현재 읽은  라인이 2개의 열로 구성되었는지/모든 열이 숫자인지 확인(소수점을 제거하고 봄)
                new_df = pd.DataFrame({'x': [float(data[0])], 'y': [float(data[1])]})  #두 조건이 모두 충족되면, 현재 라인의 데이터를 새로운 데이터프레임(new_df)으로 변환
                if not df.empty:    # 기존 데이터 프레임이 비어있지 않다면
                    df = pd.concat([df, new_df], ignore_index=True)   # 기존 데이터프레임과 새로운 데이터프레임을 연결 / 인덱스를 무시하고 새로운 인덱스를 할당하는 옵션
                else:
                    df = new_df.copy()  #비어있다면, 새로운 데이터프레임을 복사하여 df에 할당
#print(df)

#####################################
# 데이터 시각화

sb.lmplot(x='x', y='y', data=df, fit_reg=False, scatter_kws={"s" : 100})  #df에서 x,y값 각각 전달해서 산점도 그리기, 점 크기 100
plt.title('K-measns Clustering')
plt.xlabel('Mean')
plt.ylabel('SD')
#plt.show()

##############################
# k-means clustering

points = df.values #데이터 프레임 'df'의 내용을 배열로 변환해 points로 저장
kmeans = KMeans(n_clusters=3).fit(points)  #K-means 객체 생성, n_clusters 매개변수 사용해 클러스터 4개 생성, 각 데이터 포인트를 해당 클러스터에 할당
kmeans.cluster_centers_ #각 클러스터 중심 위치 구함(각 클러스터 내의 데이터 포인트 평균을 계산해 얻는 값)

kmeans.labels_ #각 데이터가 속한 클러스터 확인
df['cluster'] = kmeans.labels_ #df['cluster'] 열에는 각 데이터 포인트가 속한 클러스터의 레이블이 저장
df['label'] = ['N','N','N','N','N','N','N','N','N','N','D','D','D','D','D','D','D','D','D','D','P','P','P','P','P','P','P','P','P','P'] #라벨 추가
Result = df.groupby(['label','cluster'])['x'].count() #label과 cluster로 그룹지어줘서 각 요소가 어떤 식으로 클러스터링 되었는지 보여줌
#print(Result)
#print(df)

###############################
#최종적으로 클러스터링 완료된 결과 그래프로 출력

sb.lmplot(x='x', y='y', data=df, fit_reg=False, scatter_kws={"s":150}, hue="cluster")

def label_point(x, y, label, ax): #각 point labeling
    a = pd.concat({'x':x, 'y':y, 'label':label}, axis=1) #a라는 임시 데이터프레임 만듦
    for i, row in a.iterrows(): #한행씩 읽는
        ax.text(row['x']+.02, row['y'],  str(row['label']))   #ax.text(x좌표, y좌표, 이름문자열) : 라벨링

plt.title('K-Means clustering') 
plt.xlabel('Mean')
plt.ylabel('SD')
label_point(df.x, df.y, df.label, plt.gca())
plt.show()

###############################
# confusion matrix를 통한 성능 평가

# 레이블을 숫자로 변환(데이터 프레임엔 label이 문자로 저장되어 있지만, clustering후에는 label대신 0,1,2로 표현되기 때문에 바꾸어 주어야 함)
label_mapping = {'N': 0, 'D': 1, 'P': 2}
df['label'] = df['label'].map(label_mapping)

cm = confusion_matrix(df['label'], df['cluster'])

accuracy = accuracy_score(df['label'], df['cluster'])
precision = precision_score(df['label'], df['cluster'], average='weighted')
recall = recall_score(df['label'], df['cluster'], average='weighted')
f1 = f1_score(df['label'], df['cluster'], average='weighted')

# 결과 출력
print("K-Means clustering Score :")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)