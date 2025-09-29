from sklearn import svm
X = [[0,0,0],[2,2,2]]
y = [0.5,2.5]

clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf.fit(X,y)
print(clf.predict([[1,1,1],[2,2,2]]))