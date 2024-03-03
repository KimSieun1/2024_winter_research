from sklearn.metrics import confusion_matrix    # confusion_matrix(예측 오류값과 어떤 유형의 오류인지 나타내는 지표)라이브러리
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

##############################

# Normal : 0
# Depression : 1
# Parkinson's Disease : 2

##############################
# 각 ApEn의 mean, SD 짝꿍으로 numpy 배열 만들기

N_X = np.array([[6.4445, 0.70586],[6.9585, 0.56462],[6.8021, 0.53394],[6.7442, 0.70275],
[6.5796, 0.89296],[7.0577, 0.55136],[6.6322, 0.89268],[6.7437, 0.54025],[6.9761, 0.6008],[6.87, 0.56363]])
D_X = np.array([[6.7569, 0.69799],[6.8967, 0.68553],[7.1421, 0.47338],[6.5947, 1.0712],[7.0433, 0.68335],
[6.9897, 0.6507],[7.2871, 0.37672],[6.7796, 0.52252],[6.5843, 0.53392],[6.9796, 0.75132]])
P_X = np.array([[7.338, 0.30047],[7.1914, 0.48176],[7.0113, 0.44796],[6.8342, 0.45687],[7.1901, 0.26989],[7.2292, 0.31469],
[7.2875, 0.22427],[7.1113, 0.43081],[7.2037, 0.36894],[7.0388, 0.45183]])

##############################
# SVM 돌릴 준비

X = np.concatenate((N_X[:7], D_X[:7], P_X[:7]),axis=0) #train data
# X = np.concatenate((N_X, D_X, P_X),axis=0)
X_test = np.concatenate((N_X[7:], D_X[7:], P_X[7:]),axis=0)

y = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2] #train data
# y = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
y_test = [0,0,0,1,1,1,2,2,2]

###############################
#SVM

C = 1                         # SVM의 regularization parameter
clf= svm.SVC(kernel = "linear", C=C) #svm에서 SVC 알고리즘 가져와서 kernel linear로 설정, C=1
clf.fit(X,y) # 위에서 주어진 training set으로 margin을 최대화하는 구분선을 찾음(그 구분선을 나타내는 함수의 인자들이 조정됨)

#############################
# 훈련 결과를 시각화하는 함수 정의

# data 범위 기반 좌표축 설정
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

# 좌표축 안의 점들 target 예측 후 구분선 그리기
def plot_contours(clf, xx, yy, **params):          
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  #위에서 만든 평면 위 점들을 clf 함수에 통과시켜 classification한다. 그럼 구역별 색이 정해짐)
    Z = Z.reshape(xx.shape)     #위에서 만든 grid와 Z가 겹쳐지게 맞추기(보기쉽게)
    plt.contourf(xx, yy, Z, **params)    #나누어지는 부분에 선그리기


# 위 2개 함수 실행 및 test data의 plot 그리기
plt.figure()
X0, Y0 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, Y0)

plot_contours(clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8) #ax:그림 내 등고선 그릴 위치 지정, xx,yy : grid, alpha : 등고선 투명도)
plt.scatter(X0, Y0, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k') #test데이터 포인트를 시각화. c, cmap : 각 클래스에 따라 다른 색상 및 색상범위, 마커크기20, 마커테두리 검정)

# 그래프 환경설정
plt.xlim(xx.min(), xx.max()) #x축
plt.ylim(yy.min(), yy.max()) #y축
plt.xlabel('ApEn Mean') #x축 라벨
plt.ylabel('ApEn SD') #y축 라벨
plt.xticks(()) #x축 눈금제거
plt.yticks(()) #y축 눈금제거
plt.title('SVC with linear kernel')
plt.colorbar(label='Class')
plt.show()

##########################################
# confusion matrix를 통한 성능 평가

y_pred = clf.predict(X_test)                         # test-set X값 넣고 분류 예측

svm_cm = confusion_matrix(y_test, y_pred)                     # 예측한 Y값과 실제 정답 Y값을 비교
svm_accuracy = accuracy_score(y_test, y_pred)
svm_precision = precision_score(y_test, y_pred, average='weighted')  #weighted는 평균화 방법 중 하나, 각 클래스에 속한 셈플 수가 달라도 비교 가능하게 함
svm_recall = recall_score(y_test, y_pred, average='weighted')
svm_f1 = f1_score(y_test, y_pred, average='weighted')

print("SVM Score :")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)