#############################################################
#############################################################
#############################################################


import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    import pylab as pl

    def generate_data_set1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set2():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set3():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_hyperplane_and_support_vectors(title, clf, X_train, y_train):

        plt.title(title)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        # Plot hyperplane
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        # Plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

        plt.show()


    def run_svm_dataset1():
        X1, y1, X2, y2 = generate_data_set1()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

         ####
        # Write here your SVM code and choose a linear kernel
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####
        print("Dataset 1")

        # Linear
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM Linear")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 1 - SVM Linear", clf, X_train, y_train)

        # RBF
        clf = SVC(kernel='rbf', gamma='scale')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM RBF")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 1 - SVM RBF", clf, X_train, y_train)

        # Sigmoid
        clf = SVC(kernel='sigmoid', gamma='scale')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM Sigmoid")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 1 - SVM Sigmoid", clf, X_train, y_train)




    def run_svm_dataset2():
        X1, y1, X2, y2 = generate_data_set2()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        ####
        # Write here your SVM code and choose a linear kernel with the best C pparameter
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####
        print("Dataset 2")

        # Linear
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM Linear")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 2 - SVM Linear", clf, X_train, y_train)

        # RBF
        clf = SVC(kernel='rbf', gamma='scale')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM RBF")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 2 - SVM RBF", clf, X_train, y_train)

        # Sigmoid
        clf = SVC(kernel='sigmoid', gamma='scale')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM Sigmoid")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 2 - SVM Sigmoid", clf, X_train, y_train)



    def run_svm_dataset3():
        X1, y1, X2, y2 = generate_data_set3()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        ####
        # Write here your SVM code and use a gaussian kernel
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####
        print("Dataset 3")

        # Linear
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM Linear")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 3 - SVM Linear", clf, X_train, y_train)

        # RBF
        clf = SVC(kernel='rbf', gamma='scale')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM RBF")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 3 - SVM RBF", clf, X_train, y_train)

        # Sigmoid
        clf = SVC(kernel='sigmoid', gamma='scale')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("SVM Sigmoid")
        print("Number of instances to predict:", len(X_test))
        print("Number of instances correctly predicted:", accuracy_score(y_test, y_pred, normalize=False), "\n")

        plot_hyperplane_and_support_vectors("Dataset 3 - SVM Sigmoid", clf, X_train, y_train)



#############################################################
#############################################################
#############################################################

# EXECUTE SVM with THIS DATASETS
    run_svm_dataset1()   # data distribution 1
    run_svm_dataset2()   # data distribution 2
    run_svm_dataset3()   # data distribution 3

#############################################################
#############################################################
#############################################################
