import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn import svm
import classification as classification

from sklearn.decomposition import PCA

pca_components = 20
pca = PCA(n_components=pca_components)

# read data
lbp_features = np.load("lbp_features.npy")
num_classes = 2

# for lbp
lbp_X = lbp_features[:, :-1]
lbp_y = lbp_features[:, -1]
lbp_X = classification.standardization(lbp_X)
XtrainLBP, XtestLBP, ytrainLBP, ytestLBP = train_test_split(lbp_X, lbp_y, test_size=0.2, random_state=0)

# for hog
hog_features = np.load("hog_features.npy")
hog_X = hog_features[:, :-1]
hog_y = hog_features[:, -1]
hog_X = classification.standardization(hog_X)
XtrainHOG, XtestHOG, ytrainHOG, ytestHOG = train_test_split(hog_X, hog_y, test_size=0.2, random_state=0)

# for hist
hist_features = np.load("hist_features.npy")
hist_X = hist_features[:, :-1]
hist_y = hist_features[:, -1]
hist_X = classification.standardization(hist_X)
XtrainHIST, XtestHIST, ytrainHIST, ytestHIST = train_test_split(hist_X, hist_y, test_size=0.2, random_state=0)

# intergration
X = np.column_stack((lbp_X, hog_X, hist_X))
# 这里加上PCA
X = pca.fit_transform(X)
print(X.shape)
y = hist_y
XtrainINT, XtestINT, ytrainINT, ytestINT = train_test_split(X, y, test_size=0.2, random_state=0)

# train knn model and use cross-validation to select the best k value
# define the k value range
def optKNN():
    K = range(1, 31)
    scores = []
    std_error = []
    for k in K:
        model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        score = cross_val_score(model,  XtrainLBP, ytrainLBP, cv=5, scoring='accuracy')
        scores.append(np.array(score).mean())
        std_error.append(np.array(score).std())  
    plt.errorbar(K, scores, yerr=std_error)
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1)
    plt.show()    
    best_k = K[scores.index(max(scores))]
    print("best_k:", best_k)
    print("best_score:", scores[best_k])

# show original data and predicted data and dicision boundary of the selected knn model
best_knn_model = KNeighborsClassifier(n_neighbors=3, weights='uniform')
best_knn_model.fit(XtrainLBP, ytrainLBP)
knn_ypred = best_knn_model.predict(XtestLBP)

# train SVM model and use cross-validation to select the best C value
def optSVM():
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    scores = []
    std_error = []
    for i in range(0, len(C)):
        model = svm.LinearSVC(C=C[i])
        model.fit(XtrainLBP, ytrainLBP)
        score = cross_val_score(model, XtrainLBP, ytrainLBP, cv=5, scoring='accuracy')
        scores.append(np.array(score).mean())
        std_error.append(np.array(score).std())   

    plt.errorbar(np.log10(C), scores, yerr=std_error)
    plt.xlabel('log10(C value)')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1)
    plt.show()    
    print(scores)

best_svm_model = svm.LinearSVC(C=10)
svm_model_LBP = svm.LinearSVC(C=10)
svm_model_HOG = svm.LinearSVC(C=10)
svm_model_HIST = svm.LinearSVC(C=10)
svm_model_INT = svm.LinearSVC(C=10)

svm_model_LBP.fit(XtrainLBP, ytrainLBP)
svm_model_HOG.fit(XtrainHOG, ytrainHOG)
svm_model_HIST.fit(XtrainHIST, ytrainHIST)
svm_model_INT.fit(XtrainINT, ytrainINT)

svm_ypred_LBP = svm_model_LBP.predict(XtestLBP)
svm_ypred_HOG = svm_model_HOG.predict(XtestHOG)
svm_ypred_HIST = svm_model_HIST.predict(XtestHIST)
svm_ypred_INT = svm_model_INT.predict(XtestINT)

# define baseline classifer
base_model = DummyClassifier(strategy="most_frequent")
base_model.fit(XtrainLBP, ytrainLBP)

# start cnn

cnn_X, cnn_y = classification.compile_trainingset("/Users/wanjiang/Downloads/CMEImages/CME_polar_crop", "/Users/wanjiang/Downloads/CMEImages/NoCME_polar_crop")
cnn_X = cnn_X.astype("float32") / 255.0
cnn_xtrain, cnn_xtest, cnn_ytrain, cnn_ytest = train_test_split(cnn_X, cnn_y, test_size=0.2, random_state=0)

cnn_ytrain_1hot = keras.utils.to_categorical(cnn_ytrain, num_classes)
history, cnn_ypred = classification.CNN(cnn_xtrain, cnn_xtest, cnn_ytrain_1hot, cnn_xtest)

cm_log = ConfusionMatrixDisplay.from_predictions(cnn_ytest, cnn_ypred)
cm_log.ax_.set_title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.text(-0.1, -0.1, "True\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
plt.text(0.9, -0.1,  "False\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
plt.text(-0.1, 0.9,  "False\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
plt.text(0.9, 0.9,   "True\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
plt.show()

#confusion matrix of 3 models
def draw_confusion_matrix(model, x_data, y_data):
    cm_log = ConfusionMatrixDisplay.from_predictions(y_data, model.predict(x_data))
    cm_log.ax_.set_title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.text(-0.1, -0.1, "True\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(0.9, -0.1,  "False\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(-0.1, 0.9,  "False\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(0.9, 0.9,   "True\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.show()

def draw4FeaturesMatrices():
    draw_confusion_matrix(svm_model_LBP, XtestLBP, ytestLBP)
    draw_confusion_matrix(svm_model_HOG, XtestHOG, ytestHOG)
    draw_confusion_matrix(svm_model_HIST, XtestHIST, ytestHIST)
    draw_confusion_matrix(svm_model_INT, XtestINT, ytestINT)


def draw4ModelsMatrices():
    draw_confusion_matrix(svm_model_LBP, XtestLBP, ytestLBP)
    draw_confusion_matrix(best_knn_model, XtestLBP, ytestLBP)
    draw_confusion_matrix(base_model, XtestLBP, ytestLBP)
    # cnn matrix
    cm_log = ConfusionMatrixDisplay.from_predictions(cnn_ytest, cnn_ypred)
    cm_log.ax_.set_title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.text(-0.1, -0.1, "True\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(0.9, -0.1,  "False\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(-0.1, 0.9,  "False\nNegative", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.text(0.9, 0.9,   "True\nPositive", fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.show()

def draw4FeaturesROC():
    fpr_svm1, tpr_svm1, _ = roc_curve(ytestLBP, svm_ypred_LBP)
    plt.plot(fpr_svm1, tpr_svm1, color='r', label = "LBP feature")
    print("AUC value of SVM classifier LBP feature: ", metrics.auc(fpr_svm1, tpr_svm1))

    fpr_svm2, tpr_svm2, _ = roc_curve(ytestLBP, svm_ypred_HOG)
    plt.plot(fpr_svm2, tpr_svm2, color='r', label = "HOG feature")
    print("AUC value of SVM classifier HOG feature: ", metrics.auc(fpr_svm2, tpr_svm2))

    fpr_svm3, tpr_svm3, _ = roc_curve(ytestLBP, svm_ypred_HIST)
    plt.plot(fpr_svm3, tpr_svm3, color='r', label = "HIST feature")
    print("AUC value of SVM classifier HIST feature: ", metrics.auc(fpr_svm3, tpr_svm3))

    fpr_svm4, tpr_svm4, _ = roc_curve(ytestLBP, svm_ypred_INT)
    plt.plot(fpr_svm4, tpr_svm4, color='r', label = "Intergrated feature")
    print("AUC value of SVM classifier Intergrated feature: ", metrics.auc(fpr_svm4, tpr_svm4))
    
    plt.plot([0, 1], [0, 1], color='y',linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()

def draw4ModelsROC():
    fpr_svm, tpr_svm, _ = roc_curve(ytestLBP, svm_ypred_LBP)
    plt.plot(fpr_svm, tpr_svm, color='r', label = "SVM classifer")
    print("AUC value of SVM classifier LBP feature: ", metrics.auc(fpr_svm, tpr_svm))
    
    fpr_knn, tpr_knn, _ = roc_curve(ytestLBP, knn_ypred)
    plt.plot(fpr_knn, tpr_knn, color='g', label = "KNN classifer")
    print("AUC value of KNN classifier LBP feature: ", metrics.auc(fpr_knn, tpr_knn))
    
    fpr_cnn, tpr_cnn, _ = roc_curve(cnn_ytest, cnn_ypred)
    plt.plot(fpr_cnn, tpr_cnn, color='b', label = "CNN classifer")
    print("AUC value of CMM classifier LBP feature: ", metrics.auc(fpr_cnn, tpr_cnn))

    fpr_base, tpr_base, _ = roc_curve(ytestLBP, base_model.predict_proba(XtestLBP)[:,1])
    plt.plot(fpr_base, tpr_base, color='y', label = "Baseline classifier")
    print("AUC value of baseline classifier: ", metrics.auc(fpr_base, tpr_base))

    plt.plot([0, 1], [0, 1], color='y',linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()


#optKNN()
# optSVM()


draw4FeaturesMatrices()
draw4FeaturesROC()
#
#draw4ModelsMatrices()
draw4ModelsROC()
