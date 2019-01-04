#############################################################
#############################################################
#############################################################


import numpy as np
#import cvxopt
#import cvxopt.solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from io import StringIO
from scipy.io.arff import loadarff
import pandas as pd
from preprocessing import preprocess
from timeit import default_timer as timer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind



if __name__ == "__main__":
    import pylab as pl

    def load_data(arff_dataset_path, dataset_name):

        train_set_vec = list()
        train_labels_vec = list()
        test_set_vec = list()
        test_labels_vec = list()

        for i in range(0, 10):

            ## Nuevo
            train_path = arff_dataset_path + '/' + dataset_name + '.fold.00000' + str(i) + '.train.arff'
            with open(train_path) as file:
                content = file.read()

            data, meta = loadarff(StringIO(content))

            train_set = pd.DataFrame(data)
            train_labels = train_set.iloc[:, -1]
            train_set = train_set.iloc[:, :-1]

            test_path = arff_dataset_path + '/' + dataset_name + '.fold.00000' + str(i) + '.test.arff'
            with open(test_path) as file:
                content = file.read()

            data, meta = loadarff(StringIO(content))
            test_set = pd.DataFrame(data)
            test_labels = test_set.iloc[:, -1]
            test_set = test_set.iloc[:, :-1]

            dataset = train_set.copy()
            dataset = dataset.append(test_set, ignore_index=True)

            datalabels = train_labels.copy()
            datalabels = datalabels.append(test_labels, ignore_index=True)

            # Preprocessing and label encoding
            dataset = preprocess(dataset)
            le = LabelEncoder()
            datalabels = le.fit_transform(datalabels)

            # Getting the number of samples for training
            num_training_samples = len(train_set)

            # Separating into training and testing sets
            train_set = dataset.iloc[:num_training_samples, :]
            train_labels = datalabels[:num_training_samples]

            test_set = dataset.iloc[num_training_samples:, :]
            test_labels = datalabels[num_training_samples:]

            # Append
            train_set_vec.append(train_set)
            train_labels_vec.append(train_labels)

            test_set_vec.append(test_set)
            test_labels_vec.append(test_labels)


            ## Fin nuevo

            '''# Read train folds
            train_path = arff_dataset_path + '/' + dataset_name + '.fold.00000' + str(i) + '.train.arff'
            with open(train_path) as file:
                content = file.read()

            data, meta = loadarff(StringIO(content))

            train_set = pd.DataFrame(data)
            train_labels = train_set.iloc[:, -1]
            train_set = train_set.iloc[:, :-1]

            if dataset_name == 'hypothyroid':
                del train_set['TBG']

            train_set = preprocess(train_set)

            train_set_vec.append(train_set)
            le.fit(train_labels)
            train_labels = le.transform(train_labels)
            train_labels_vec.append(train_labels)

            # Read test folds
            test_path = arff_dataset_path + '/' + dataset_name + '.fold.00000' + str(i) + '.test.arff'
            with open(test_path) as file:
                content = file.read()

            data, meta = loadarff(StringIO(content))
            test_set = pd.DataFrame(data)
            test_labels = test_set.iloc[:, -1]
            test_set = test_set.iloc[:, :-1]
            test_set = preprocess(test_set)

            if dataset_name == 'hypothyroid':
                del test_set['TBG']

            test_set_vec.append(test_set)
            test_labels = le.transform(test_labels)
            test_labels_vec.append(test_labels)'''

        return train_set_vec, train_labels_vec, test_set_vec, test_labels_vec

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


    # Exercise 2 functions
    def run_svm_exercise2_svm_linear(datasets):

        best_avg_acc = 0
        best_avg_time = 0
        best_avg_c = 0

        # For each dataset
        for dataset in datasets:

            X_train, y_train, X_test, y_test = load_data(dataset[0], dataset[1])

            # All possible values of C
            for c in np.arange(0.1, 1.1, 0.1):

                # Linear SVM
                clf = SVC(kernel='linear', C=c)

                # Attributes
                linear_avg_acc = 0
                linear_avg_time = 0

                # K-Fold Cross Validation
                K = 10

                for i in range(0, K):

                    # Start timer
                    start_time = timer()

                    # Training
                    clf.fit(X_train[i], y_train[i])

                    # Stop timer
                    end_time = timer()

                    # Predicting
                    y_pred = clf.predict(X_test[i])

                    # Saving the time and accuracy
                    acc = accuracy_score(y_test[i], y_pred)
                    time = end_time - start_time

                    print("{0}, linear, fold={1}, C={2}, acc={3}, time={4}".format(dataset[1], i, clf.C, acc, time))

                    linear_avg_acc += acc
                    linear_avg_time += time


                # Average values
                linear_avg_acc /= K
                linear_avg_time /= K

                # Update winner
                if (linear_avg_acc > best_avg_acc):

                    best_avg_acc = linear_avg_acc
                    best_avg_time = linear_avg_time
                    best_avg_c = clf.C


                # Print values
                #print("{0} Average, linear, C={1}, acc={2}, time={3}".format(dataset[1], clf.C, linear_avg_acc, linear_avg_time))

            # Best for this dataset
            print("Best hyperparameters: {0}, Linear, C={1}, acc={2}, time={3}".format(dataset[1], best_avg_c, best_avg_acc, best_avg_time))

            # Clean values
            best_avg_acc = 0
            best_avg_time = 0
            best_avg_c = 0

    def run_svm_exercise2_svm_rbf(datasets):

        best_avg_acc = 0
        best_avg_time = 0
        best_avg_c = 0
        best_avg_gamma = 0

        # For each dataset
        for dataset in datasets:

            X_train, y_train, X_test, y_test = load_data(dataset[0], dataset[1])

            # All possible values of C
            for c in np.arange(0.1, 1.1, 0.1):

                # All possible values of gamma
                for g in np.arange(0.1, 1.1, 0.1):

                    # Linear SVM
                    clf = SVC(kernel='rbf', C=c, gamma=g)

                    # Attributes
                    rbf_avg_acc = 0
                    rbf_avg_time = 0

                    # K-Fold Cross Validation
                    K = 10

                    for i in range(0, K):
                        # Start timer
                        start_time = timer()

                        # Training
                        clf.fit(X_train[i], y_train[i])

                        # Stop timer
                        end_time = timer()

                        # Predicting
                        y_pred = clf.predict(X_test[i])

                        # Saving the time and accuracy
                        acc = accuracy_score(y_test[i], y_pred)
                        time = end_time - start_time

                        print("{0}, rbf, fold={1}, C={2}, Gamma={5}, acc={3}, time={4}".format(dataset[1], i, clf.C, acc, time, clf.gamma))

                        rbf_avg_acc += acc
                        rbf_avg_time += time

                    # Average values
                    rbf_avg_acc /= K
                    rbf_avg_time /= K

                    # Update winner
                    if (rbf_avg_acc > best_avg_acc):

                        best_avg_acc = rbf_avg_acc
                        best_avg_time = rbf_avg_time
                        best_avg_c = clf.C
                        best_avg_gamma = clf.gamma

                    # Print values
                    #print("{0} Average, rbf, C={1}, Gamma={4}, acc={2}, time={3}".format(dataset[1], clf.C, rbf_avg_acc, rbf_avg_time, clf.gamma))

            # Best for this dataset
            print("Best hyperparameters: {0}, RBF, C={1}, Gamma={2}, acc={3}, time={4}".format(dataset[1], best_avg_c, best_avg_gamma, best_avg_acc, best_avg_time))

            # Clean values
            best_avg_acc = 0
            best_avg_time = 0
            best_avg_c = 0
            best_avg_gamma = 0


    def run_svm_exercise2_svm_vedaldi(datasets):

        best_avg_acc = 0
        best_avg_time = 0
        best_avg_c = 0

        # For each dataset
        for dataset in datasets:

            X_train, y_train, X_test, y_test = load_data(dataset[0], dataset[1])

            # All possible values of C
            for c in np.arange(0.1, 1.1, 0.1):

                clf = SVC(kernel='precomputed', C=c)

                # Attributes
                vedaldi_avg_acc = 0
                vedaldi_avg_time = 0

                # K-Fold Cross Validation
                K = 10

                for i in range(0, K):
                    # Start timer
                    start_time = timer()

                    # Precompute train kernel
                    kernel_train_matrix = np.zeros((X_train[i].shape[0], X_train[i].shape[0]))

                    for x in range(0, kernel_train_matrix.shape[0]):

                        vector_x = X_train[i].iloc[x]

                        for y in range(0, kernel_train_matrix.shape[0]):

                            sum = 0

                            vector_y = X_train[i].iloc[y]

                            denominador = np.sum(vector_x) + np.sum(vector_y)

                            if (denominador > 0):
                                nominador = 2 * np.dot(vector_x, vector_y)
                                sum += nominador / denominador

                            kernel_train_matrix[x][y] = sum
                    # End of precomputing training kernel

                    # Precompute test kernel
                    kernel_test_matrix = np.zeros((X_test[i].shape[0], X_train[i].shape[0]))

                    for x in range(0, kernel_test_matrix.shape[0]):

                        vector_x = X_test[i].iloc[x]

                        for y in range(0, kernel_test_matrix.shape[1]):

                            sum = 0

                            vector_y = X_train[i].iloc[y]

                            denominador = np.sum(vector_x) + np.sum(vector_y)

                            if (denominador > 0):
                                nominador = 2 * np.dot(vector_x, vector_y)
                                sum += nominador / denominador

                            kernel_test_matrix[x][y] = sum
                    # End of precomputing testing kernel

                    # Training
                    clf.fit(kernel_train_matrix, y_train[i])

                    # Stop timer
                    end_time = timer()

                    # Predicting
                    y_pred = clf.predict(kernel_test_matrix)

                    # Saving the time and accuracy
                    acc = accuracy_score(y_test[i], y_pred)
                    time = end_time - start_time

                    print("{0}, Vedaldi, fold={1}, C={2}, acc={3}, time={4}".format(dataset[1], i, clf.C, acc, time))

                    vedaldi_avg_acc += acc
                    vedaldi_avg_time += time

                # Average values
                vedaldi_avg_acc /= K
                vedaldi_avg_time /= K

                # Update winner
                if (vedaldi_avg_acc > best_avg_acc):

                    best_avg_acc = vedaldi_avg_acc
                    best_avg_time = vedaldi_avg_time
                    best_avg_c = clf.C

                # Print values
                # print("{0} Average, linear, C={1}, acc={2}, time={3}".format(dataset[1], clf.C, linear_avg_acc, linear_avg_time))

            # Best for this dataset
            print("Best hyperparameters: {0}, Vedaldi and Zisserman, C={1}, acc={2}, time={3}".format(dataset[1], best_avg_c, best_avg_acc, best_avg_time))

            # Clean values
            best_avg_acc = 0
            best_avg_time = 0
            best_avg_c = 0


    def doStatisticalComparison():
        Linear = [0.8605470075354085, 0.8801134333379176, 0.850718462823726, 0.8517507002801121]
        RBF = [0.8709885142054372, 0.9088241937017386, 0.8561199169093905, 0.8396778711484594]
        Vedaldi = [0.8548327475509494, 0.8736361794494065, 0.8398325769378401, 0.8342121848739495]

        T1, p1 = ttest_ind(RBF, Linear)
        # interpret
        alpha = 0.05
        if p1 > alpha:
            print('RBF, Linear - Data is not statistically different')
        else:
            print('RBF, Linear - Data is statistically different')
        print(p1)
        T2, p2 = ttest_ind(RBF, Vedaldi)
        if p2 > alpha:
            print('RBF, Vedaldi - Data is not statistically different')
        else:
            print('RBF, Vedaldi - Data is statistically different')
        print(p2)
        T3, p3 = ttest_ind(Linear, Vedaldi)
        if p3 > alpha:
            print('Linear, Vedaldi - Relief is not statistically different')
        else:
            print('Linear, Vedaldi - Relief is statistically different')
        print(p3)


#############################################################
#############################################################
#############################################################

# EXECUTE SVM with THIS DATASETS - Exercise 1
    #run_svm_dataset1()   # data distribution 1
    #run_svm_dataset2()   # data distribution 2
    #run_svm_dataset3()   # data distribution 3

#############################################################
#############################################################
#############################################################


#############################################################
#############################################################
#############################################################

# EXECUTE SVM with TWO CBR DATASETS - Exercise 2

    datasets = [
        ['./datasetsCBR/credit-a', 'credit-a'],
        #['./datasetsCBR/bal', 'bal']
    ]
    #run_svm_exercise2_svm_linear(datasets)          # SVM Linear
    #run_svm_exercise2_svm_rbf(datasets)             # SVM RBF
    #run_svm_exercise2_svm_vedaldi(datasets)         # SVM Linear splines
    doStatisticalComparison()

#############################################################
#############################################################
#############################################################