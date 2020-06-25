from itertools import product as iterproduct

from pylab import *;ion()
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

print('[INFO]: Loading data')
data_dict = joblib.load('simulated_bad_pixels_df.joblib.save')

print('[INFO]: Selecting data')
features = data_dict['features']
labels = data_dict['labels']

print('[INFO]: Train Test Splitting')
idx_train, idx_test = train_test_split(np.arange(labels.size), 
                                        test_size=0.2,
                                        stratify=labels)

print('[INFO]: Transforming with PCA')
std_sclr = StandardScaler()
pca = PCA()
features_std_train = std_sclr.fit_transform(features[idx_train])
features_std_test = std_sclr.transform(features[idx_test])

features_pca_train = pca.fit_transform(features_std_train)
features_pca_test = pca.transform(features_std_test)

print('[INFO]: Establishing Random Forest Classifier')
rfr = RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1)

print('[INFO]: Fitting Random Forest Classifier')
rfr.fit(features_pca_train, labels[idx_train])
print('[INFO]: Finished Fitting Random Forest Classifier')

print('[INFO]: Predicting Random Forest Classifier')
predict = rfr.predict(features_pca_test)

print('[INFO]: Computing Quality Metrics')
oob_score = rfr.oob_score_
accuracy = accuracy_score(labels[idx_test], predict)

num_per_class = [(labels[idx_test] == label).sum() for label in set(labels)]
confusionMatrix = confusion_matrix(labels[idx_test], predict) / num_per_class

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in iterproduct(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

class_names = ['Clean', 'Hot Pixel', 'Cold Pixel', 'Sat Hot Pixel', 'Sat Cold Pixel', 
                'Cosmic Ray', 'Popcorn Pixel', 'Noisy']
plot_confusion_matrix(confusionMatrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

fig = gcf()
fig.savefig('RandomForest300_ConfusionMatrix.png')

gotItRight = labels[idx_test] == predict

class_GIR = {classnow:np.where((labels[idx_test] == classnow)*gotItRight)[0][0] \
                for classnow in set(labels) \
                if np.any((labels[idx_test] == classnow)*gotItRight)}

class_nGIR = {classnow:np.where((labels[idx_test] == classnow)*(~gotItRight))[0][0] \
                for classnow in set(labels) \
                if np.any((labels[idx_test] == classnow)*(~gotItRight))}

# Got It Right
for key, val in class_GIR.items(): 
    # fig = figure()
    plt.plot(features[idx_test][val], label=class_names[predict[val]])
    
    legend(loc=4, fontsize=20, framealpha=0.9)
    
    xlim(0,110)
    ylabel('Electrons Read Off Detetor', fontsize=30)
    xlabel('Group Number [0,107]', fontsize=30)
    legend(loc=4, fontsize=15, framealpha=.9)
    title('Correctly Predicted', fontsize=30)
    
    ax = gcf().get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    
    fig.savefig('Raw_data_correct_prediction.png')
    # fig.savefig('Raw_data_correct_prediction_{}.png'.format(class_names[predict[val]]))

# not Got It Right
for key, val in class_nGIR.items(): 
    fig = figure()
    plt.plot(features[idx_test][val], label=class_names[predict[val]])
    legend(loc=0, fontsize=20, framealpha=0.9)
    
    xlim(0,110)
    ylabel('Electrons Read Off Detetor', fontsize=30)
    xlabel('Group Number [0,107]', fontsize=30)
    legend(loc=4, fontsize=15, framealpha=.9)
    title('Incorrectly Predicted: {}'.format(class_names[predict[val]]), fontsize=30)
    
    ax = gcf().get_axes()[0]
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    
    fig.savefig('Raw_data_wrong_prediction_{}.png'.format(class_names[predict[val]]))