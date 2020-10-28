#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:37:35 2017

@author: peter
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/home/peter/gdrive/dev/Functions')
sys.path.append('F:\\GoogleDrive\\dev\\Functions')
import FUNCTIONS_ARRAYS as fa

import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D #Dont rmeove
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import os
#%%
# Load dat Training and classification Data
#Input
path = 'H:\\sylt_inken\\' 
train_or_class = 'class'
training = 'test_files_attributed.pckl'
classif = 'test_files_attributed.pckl'
franz_grund_truth_file = '/home/peter/databox/NeuralSeafloor/GroundTruth/tabellen_mit_median/proben_medisort_folk_figge.csv'             # tab seperated csv
sup_unsup = 'unsup'   #sup or unsup  for classification?

#Output of results as ASCII (mode = classification)

#Scaler for Data normalization
scaler = MinMaxScaler()   #StandardScaler MinMaxScaler RobustScaler  http://benalexkeen.com/feature-scaling-with-scikit-learn/
#scaler = RobustScaler(quantile_range = (5,95))
#PCA 
PCA_anal = 'no'
PCA_components = 3    # set two 2 and 3 will generate plots of kmeans if unsup


iow_n_only = 'no'  #Use only high quality iow_n data


################################################''TRAINING AND GROUND TRUTHING'############################
# Parameter Tuning
parameter_tuning = 'no'


#Feature selection
brute_force_feature_selection = 'no' #using the supervised classification
score = 'f1_weighted' #<- used to account for class imbalance: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

# Confusion Matrix and plot of decision function
cf_matrix= 'yes'

#Klassifikations_level
label = ""

#Features to use from input pandas df
#features = ["BSmed", "BSstd", "bathy", "diffme200", "tri200", "std200", "std4km" ]
features = ["180avg_1"]
#features = ["BSmed", "std200" ,"bathy"]

# Klassen gruppieren?
group_sediments = 'no'
group_classes = {'FSed': "A", 'S': "A", 'MxSed' : 'A', 'CSed' : 'A', 'LagSed' : 'A'}

minimum_number_of_elements_per_class = 5   #removes all clases with less than this number of elements
clean_from_placeholders = 'yes' # removes lines with entries "1" and "NC

####kmeans plots
# Allgemein
range_n_clusters = [4,5,6]   
# Compare cluster methods
compare_cluster_methods = 'no'

#Modul Training
plot_kmeans ='no'    #unsupervised classification if PCA_components =2 or =3
n_clusters = 4  #for plot_kmeans only
use_numbers_of_labels_for_clusters_instead = 'no'
plot_shilouette_plot = 'no'
compare_unsup_data_with_ground_truth = 'yes' # compare with franz ground-trtuh data and export model file for plotting program. Uses range_n_clusters
kmeansoutname = 'test.csv'

###Modul Klassifikation
make_kmeans_grid = 'yes'   #Gridde den kmeans output  # gridde nur mit UTM Koordinaten, muss Feld X/Y geben
make_sup_grid = 'no'   #Gridde den kmeans output  # gridde nur mit UTM Koordinaten, muss Feld X/Y geben
grid_resolution = 60
 #schreibe die klassifizierungsdaten und die grids
out = 'test2.txt'   #class data
out_grid_base = 'testgrid'
grid_search_radius = 5








################################################'MACHINE LEARNING PARAMETER SETTING##########################
# Names and parameters of classifiers. Be super-careful with the ordering of 
# the lists!
names = [ "RBF_SVM"
         ]

#Ideal classifiers for Ebene A
if label == "ebene_a":
    print "Verwende classifier für ebene_a"
    classifiers = [
        SVC(kernel = 'rbf' , gamma=0.1, C=1, class_weight = 'balanced'),
        #RandomForestClassifier(max_features = 'auto', class_weight = 'balanced', min_samples_split = 4, criterion = 'gini', n_estimators = 100),
        ]
#Ideal classifiers for Ebene B
elif label == "ebene_b":
     print "Verwende classifier für ebene_b"
     classifiers = [
        SVC(kernel = 'rbf' , gamma=0.000001, C=1000,class_weight = 'balanced'),
        RandomForestClassifier(max_features = 'auto', class_weight = 'balanced_subsample', min_samples_split = 100, criterion = 'entropy', n_estimators = 40),
        ]
elif label == "ebene_c":
     print "Verwende classifier für ebene_c"
     classifiers = [
        SVC(kernel = 'rbf' , gamma=0.000001, C=1000, class_weight = 'balanced'),
        RandomForestClassifier(max_features = 'auto', class_weight = 'balanced', min_samples_split = 60, criterion = 'entropy', n_estimators = 40),
        ]
else:
    print "Verwende Standard Classifier-Settings: Keine spezial-settings gefunden"
    classifiers = [
    SVC(kernel = 'rbf' , gamma=0.001, C=1, class_weight = 'balanced'),
    RandomForestClassifier(max_features = 'auto', class_weight = 'balanced', min_samples_split = 0.05, criterion = 'entropy', n_estimators = 20),
    ]
#############################################################################
# Parameters for hyperparameter-search
#Tune the hyper parameters. Das Programm bricht dann nach dem tunen ab
# Ergebnisse werden in die Datei logfile in path geschrieben


classifiers_to_tune = [SVC
                       ]

#Name of variable: classifier+tuned_parameters
# Tune SVM
SVC_tuned_parameters = [{
                    'kernel': ['linear', 'rbf', 'sigmoid'], 
                     'gamma': [0,0000001, 0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                     'C': [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
                     'class_weight' : ['balanced']
                 }]

#Tune Random Forests
RandomForestClassifier_tuned_parameters = [{
                    'n_estimators': [4 ,30, 50, 75, 100], 
                     'criterion': ['gini', 'entropy'],
                     'min_samples_split' : [2,4,8,10,20,50, 100],
                     'class_weight' : ['balanced', 'balanced_subsample']
                 }]




classifier_parameters_to_tune = [SVC_tuned_parameters
                                 ]


#%% Functrions

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
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
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def merge_idx_stat(other_data, kmeanslabel, idxstat, n_clusters):
    #kmeanslabel is the array coming out of the sklearn kmeans
    # idxstat are the labels from the original data file
    df= pd.DataFrame(data=kmeanslabel, columns=['Cluster_n_' + str(n_clusters)])
    #print df.head()
    df['idx_stat'] = idxstat
    #print df.head()
    result = pd.merge(other_data, df, how = 'outer', on = ['idx_stat'])


    #print model_data.head()
    return result

def merge_stat(other_data, kmeanslabel, n_clusters):
    #kmeanslabel is the array coming out of the sklearn kmeans
    # idxstat are the labels from the original data file
    df= pd.DataFrame(data=kmeanslabel, columns=['Cluster_n_' + str(n_clusters)])
    #print df.head()
    #print df.head()
    class_data = pd.merge(other_data, df, how = 'outer', on = ['idx_stat'])


    #print model_data.head()
    return class_data


def plot_decision_function(X,Y,clf):
    from mlxtend.plotting import plot_decision_regions
    plot_decision_regions(X,Y,clf=clf, res = 0.005, legend=2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim((0,1))
    plt.ylim((0,1))


def write_asc_grd(filepath, grid_x, grid_y, resolution, data):
    
    nrows = data.shape[0]
    ncols = data.shape[1]
    xllcorner = grid_x.min()
    yllcorner = grid_y.min()
    

    line1 = "NCOLS " + str(ncols)
    line2 = "NROWS "+ str(nrows)
    line3 = "XLLCORNER " + str(xllcorner,)
    line4 = "YLLCORNER " + str(yllcorner)
    line5 = "NODATA_value -9999"

    #Create header
    header='\n'.join([line1, line2, line3, line4, line5])
    #Wrtie file
    np.savetxt(filepath, data, header=str(header), comments='')
    return

#%%###########################################################################
    #Datenaufbereitung

try:
    model_data = pd.read_pickle(path + training)
    if iow_n_only == 'yes':
        model_data = model_data[model_data['quelle'] == 'iow_n']
    # Remove nan entries
    print "Dropping NAN lines"
    model_data = model_data.dropna(subset = [features])
    
    print "Filtering classes with less than " , minimum_number_of_elements_per_class, " occurences"
    number_of_occurence = model_data.groupby(label).aggregate(np.count_nonzero)
    tags = number_of_occurence[number_of_occurence.idx_stat >= minimum_number_of_elements_per_class].index   # idx_stat nur als Beispiel, alle Kolumnen kommen gleich oft vor
    model_data = model_data[model_data[label].isin(tags)]
     
    #####clean from placeholders
    if clean_from_placeholders == 'yes':
        print "Removing entries with label 1 and NC"
        model_data = model_data[model_data[label] != '1']
        model_data = model_data[model_data[label] != 'NC']

        
    #Standardize Data
    for column in model_data[features]:
        model_data[[column]] = scaler.fit_transform(model_data[[column]])
    
    #retrieve idx_stat
    idxstat = model_data['idx_stat']
except:
    print('Keine Trainingsdatensdaten vorhanden')
    pass

try:
    class_data = pd.read_pickle(path + classif)
    #Remove NaN entries
    class_data = class_data.dropna(subset = [features])
    class_data_present ='yes'
    if len(class_data) == 0:
        print "Warning: Class Data has 0 rows - wrong features selected or incorrectly spelled?"
        print "Setting clasas_data present to NO"
        class_data_present = 'yes'
except:
    print('Keine Klassifizierungsdaten vorhanden. Fahre fort mit Training')
    class_data_present = 'no'
    
if class_data_present == 'yes':
    for column in class_data[features]:
         class_data[[column]] = scaler.fit_transform(class_data[[column]])
    
#Eventuelle Gruppierung durchführen
if group_sediments == 'yes':
    print "Grouping sediments"
    model_data[label].replace(group_classes, inplace=True)

try:
     pd.value_counts(model_data[label]).plot.bar(title='Counts per unique label class in model data')
except:
     print "Cannot plot number of unqiue labels classes - classif. data only?"

#Convert categorcial labels to numerical labels
try:
    class_names_string = model_data[label]
    model_data, label_dict = fa.convert_from_categorical(model_data, label)
    label_dict_inv = {v: k for k, v in label_dict.iteritems()} # fur rücktransofmration
except:
    print "Keine Klassennamen - vermute nur Klassifikationsdaten"
    pass

# Add results columns to original data frame
if class_data_present == 'yes':
    for name in names:
        class_data[name] = np.nan

#Auswahl der Parameter und IMputer

model_X = model_data.as_matrix(features)
model_y = model_data.as_matrix([label])
model_y = np.ravel(model_y)

if class_data_present == 'yes':
    try:
        class_X = class_data.as_matrix(features)
    except:
        print 'Kann Klassifizierungsdaten nicht in MAtrix konvertieren'


#%%###########################################################################
#PCA
if PCA_anal == 'yes':
    print 'PCA of train data'
    pca = PCA(n_components=PCA_components)
    model_X = pca.fit_transform(model_X)
    print pca.explained_variance_ratio_
    print pd.DataFrame(pca.components_,columns=features)

    try:
        print 'PCA of class data'
        class_X = pca.fit_transform(class_X)
        #print pca.explained_variance_ratio_
        
        #print pca.components_
    except:
        print "Klassifikationsdaten für PCA nicht eingeladen"


#%%###########################################################################
# Parameter search
if parameter_tuning == 'yes':
    print "Starte Parameter Tuning"
    print "Using score :" , score
    #Use kfold to increase number of samples for hyperparameter estimation
    cv = KFold(n_splits=5, shuffle = True)
    logfile = 'logfile.txt'
    log = open(path+logfile, "a")
    i = 0
    print 'Klassifiziere für ', path
    for classifier_to_tune in classifiers_to_tune:
        print classifier_to_tune
        print classifier_to_tune
        print "# Tuning hyper-parameters for %s" % score
        print  ()
        # Set the parameters by cross-validation
        clf = GridSearchCV(classifier_to_tune(), classifier_parameters_to_tune[i], cv=5,
                           scoring=score)
        #clf.fit(X_train, y_train)
        clf.fit(model_X, model_y)
        print ("Best parameters set found on development set:")
        print ()
        print (clf.best_params_) 
        print (clf.best_score_)
        #print ()
        #print ("Grid scores on development set:")
        #print ()
        #means = clf.cv_results_['mean_test_score']
        #stds = clf.cv_results_['std_test_score']
        #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #    print ("%0.3f (+/-%0.03f) for %r"
        #          % (mean, std * 2, params))
        print ()
        i = i + 1  
        
        
        # combine with optimal feature selection: see example 5
        # http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/#example-5-exhaustive-feature-selection-and-gridsearch
    log.close()  
#%%

if brute_force_feature_selection == 'yes':
    print "Determining optimal features"
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
    for classifier in classifiers:
        print "Feature determination for classifier:", classifier
        efs1 = EFS(classifier, 
                   min_features=1,
                   max_features=len(features),
                   scoring='accuracy',
                   print_progress=True,
                   cv=15)
        efs1 = efs1.fit(model_X, model_y)
        print('Best accuracy score: %.2f' % efs1.best_score_)
        print('Best subset:', efs1.best_idx_)
        feat_extr = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
        feat_extr.sort_values('avg_score', inplace=True, ascending=False)
        print feat_extr
        
        metric_dict = efs1.get_metric_dict()

        fig = plt.figure()
        fig.set_size_inches(20, 6)
        k_feat = sorted(metric_dict.keys())
        avg = [metric_dict[k]['avg_score'] for k in k_feat]
        
        upper, lower = [], []
        for k in k_feat:
            upper.append(metric_dict[k]['avg_score'] +
                         metric_dict[k]['std_dev'])
            lower.append(metric_dict[k]['avg_score'] -
                         metric_dict[k]['std_dev'])
        
        plt.fill_between(k_feat,
                         upper,
                         lower,
                         alpha=0.2,
                         color='blue',
                         lw=1)
        
        plt.plot(k_feat, avg, color='blue', marker='o')
        plt.ylabel('Accuracy +/- Standard Deviation')
        plt.xlabel('Number of Features')
        feature_min = len(metric_dict[k_feat[0]]['feature_idx'])
        feature_max = len(metric_dict[k_feat[-1]]['feature_idx'])
        plt.xticks(k_feat, 
                   [str(metric_dict[k]['feature_idx']) for k in k_feat], 
                   rotation=90)
        plt.show()
        
    
#%%
# Make a confusion matrix with the chosen parameters
if cf_matrix == 'yes':
    from sklearn.metrics import classification_report
    print "Plotte Confusion Matrix"
    for classifier in classifiers:
        """
        labels : array, shape = [n_classes], optional List of labels to index the matrix. 
        This may be used to reorder or select a subset of labels. If none is given, those that appear at 
        least once in y_true or y_pred are used in sorted order.
        """
        print classifier
        X_train, X_test, y_train, y_test = train_test_split(model_X, model_y, random_state=42, train_size=0.75)
        y_pred = classifier.fit(X_train, y_train).predict(X_test)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        # Plot non-normalized confusion matrix
        #plt.figure()
        #plot_confusion_matrix(cnf_matrix, classes=label_dict,
         #             title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.figure()
        class_names = np.unique(y_test)
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                             title='Confusion matrix')
        
        plt.show()
        """
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives and 
        fp the number of false positives. The precision is intuitively the ability of the classifier 
        not to label as positive a sample that is negative.
        The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the 
        number of false negatives. The recall is intuitively the ability of the classifier 
        to find all the positive samples.
        The F-beta score can be interpreted as a weighted harmonic mean of the 
        precision and recall, where an F-beta score reaches its best value at 
        1 and worst score at 0.
        The F-beta score weights recall more than precision by a factor of beta.
        beta == 1.0 means recall and precision are equally important.
        The support is the number of occurrences of each class in y_true.
"""
        
        #print "PLotte classification Report"
        #print(classification_report(y_test, y_pred, target_names=class_names))
        
        
        # Feature importances
        try:
            imp = classifier.feature_importances_
            if PCA_anal =='no':
                names = features
            else:
                names = np.arange(0,PCA_components)
            
            imp, names = zip(*sorted(zip(imp, names)))
            plt.barh(range(len(names)), imp, align='center')
            plt.yticks(range(len(names)),names)
            plt.xlabel('Feature importance')
            plt.ylabel('Features')
            plt.title('Importance of each feature')
            plt.show()
        except:
            pass


        try:
            plot_decision_function(model_X, model_y, classifier)
        except:
            pass
#%%
if compare_cluster_methods == 'yes':
    plt.close()
    import time
    import warnings
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn import cluster, datasets, mixture
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from itertools import cycle, islice
    

    
    X, y = model_X, model_y
    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(15, 9))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    
    plot_num = 1
    for n_clusters in range_n_clusters:


        
        default_base = {'quantile': .3,
                        'eps': .3,
                        'damping': .9,
                        'preference': -200,
                        'n_neighbors': 10,
                        'n_clusters': n_clusters}
    
        
     
        # update parameters with dataset-specific values
        params = default_base.copy()
    
    
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
    
        # ============
        # Create cluster objects
        # ============
    
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    
        spectral_rbf = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="rbf")
        
        spectral_near = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')
    
        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('SpectralClustering_rbf', spectral_rbf),
            ('SpectralClustering_neanrn', spectral_near),
            ('GaussianMixture', gmm)
        )
    
        for name, algorithm in clustering_algorithms:
            t0 = time.time()
    
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)
    
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
    
            plt.subplot(len(range_n_clusters), len(clustering_algorithms), plot_num)
            plt.title((name, n_clusters), size=10)
    
            #color = cm.spectral(float(i) / n_clusters)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                             int(max(y_pred) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1
    
    plt.show()

#%%
if plot_kmeans == 'yes':
        print 'Attempting to plot cluster of model_X data in2D or 3D'
        if (PCA_components == 2 and PCA_anal == 'yes'):
            print 'Plotting kmeans with PCA=2'
            if use_numbers_of_labels_for_clusters_instead == 'yes':
                n_clusters = len(np.unique(model_y))
            #Create Tests for comparison with ture labels
            kmeans = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100,
                              n_init=10, max_no_improvement=10, verbose=0)
            kmeans.fit(model_X)
            
            # Step size of the mesh. Decrease to increase the quality of the VQ.
            h = 0.02     # point in the mesh [x_min, x_max]x[y_min, y_max].
            
            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = model_X[:, 0].min() - 0.1, model_X[:, 0].max() + 0.1
            y_min, y_max = model_X[:, 1].min() - 0.1, model_X[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Obtain labels for each point in mesh. Use last trained model.
            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
            
            # Put the result into a color plot
        
            Z = Z.reshape(xx.shape)
            plt.figure(1, figsize=(10,8))
            plt.clf()
            plt.imshow(Z, 
                       interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Paired,
                       aspect='auto', 
                       origin='lower')

        
            
            plt.plot(model_X[:, 0], 
                     model_X[:, 1], 
                     'k.', 
                     markersize=5,
                     c='black',
                     label = model_y
                     )
            # Plot the centroids as a white X
            centroids = kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], 
                        centroids[:, 1],
                        marker='x', 
                        s=169, 
                        linewidths=3,
                        c='w',
                        zorder=10)
            plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                      'Centroids are marked with white cross')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            
        elif PCA_components == 3:
            # http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html check BSD license
            if use_numbers_of_labels_for_clusters_instead == 'yes':
                n_clusters = len(np.unique(model_y))
            print 'Plotting kmeans with PCA=3'
            kmeans = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100,
                              n_init=10, max_no_improvement=10, verbose=0)
            kmeans.fit(model_X)
            
            # PLot the kmeans cluster
            fig, (ax3) = plt.subplots(1, 1)
            fig.set_size_inches(8, 6)
            ax3 = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=60, azim=60)
            labels = kmeans.labels_
            ax3.scatter(model_X[:, 0], 
                       model_X[:, 1], 
                       model_X[:, 2],
                       c=labels.astype(np.float), edgecolor='k')
            ax3.w_xaxis.set_ticklabels([])
            ax3.w_yaxis.set_ticklabels([])
            ax3.w_zaxis.set_ticklabels([])
            ax3.set_xlabel('PCA1')
            ax3.set_ylabel('PCA2')
            ax3.set_zlabel('PCA3')
            ax3.set_title('Kmeans')
            ax3.dist = 12
        else:
            print 'Cannot plot kmeans. Set PCA-anal to yes and PCA_components to 2 or 3 to plot'
    
if plot_shilouette_plot == 'yes':
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(6, 4)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(model_X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(model_X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(model_X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(model_X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
       
        # 2nd Plot showing the actual clusters formed in 2D
        """
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(model_X[:, 0], 
                    model_X[:, 1], 
                    marker='.', 
                    s=30, 
                    lw=0, 
                    alpha=0.7,
                    c=colors, 
                    edgecolor='k')
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], 
                    centers[:, 1], 
                    marker='o',
                    c="white", 
                    alpha=1, 
                    s=200, 
                    edgecolor='k')
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], 
                        c[1], 
                        marker='$%d$' % i, 
                        alpha=1,
                        s=50, 
                        edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, 
                     fontweight='bold')
        
        """
        fig, (ax2) = plt.subplots(1, 1)
        fig.set_size_inches(10, 8)
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2 = Axes3D(fig, 
                     rect=[0, 0, 0.95, 1], 
                     elev=60, 
                     azim=60
                     )
        ax2.scatter(model_X[:, 0], 
                   model_X[:, 1], 
                   model_X[:, 2],
                   c=colors, 
                   edgecolor='k')
        ax2.w_xaxis.set_ticklabels([])
        ax2.w_yaxis.set_ticklabels([])
        ax2.w_zaxis.set_ticklabels([])
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        ax2.set_ylabel("Feature space for the 3rd feature")
        ax2.dist = 12
        # Labeling the clusters
        centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], 
                    centers[:, 1],
                    centers[:, 2], 
                    marker='o',
                    c="white", 
                    alpha=1, 
                    s=200, 
                    edgecolor='red', 
                    linewidth=3)
        for i, c in enumerate(centers):
                ax2.scatter(c[0], 
                        c[1],
                        c[2], 
                        marker='$%d$' % i, 
                        alpha=1,
                        s=50, edgecolor='k')
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                  fontsize=14, 
                  fontweight='bold')
        
        # speichern?
#%%
        
    #%%
if compare_unsup_data_with_ground_truth == 'yes':
    # Do the clustering    
    for n_clusters in range_n_clusters:
            print 'Kkmeans with cluster number: ', n_clusters
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(model_X)
            print 'Add results to datafile model_data on idx_stat'
            model_data = merge_idx_stat(model_data, cluster_labels, idxstat, str(n_clusters))
    # read franz data
    print "Read Franz data"
    franz_df = pd.read_csv(franz_grund_truth_file, sep='\t')
    franz_df = franz_df.drop(['ebene_a', 'ebene_b', 'ebene_c', "quelle"], axis=1)  #die gbts schon
    model_data = pd.merge(model_data, franz_df, how='inner', on = ['idx_stat'])
    
    #Labelnamen wiederherstellen
    model_data = model_data.replace({label:label_dict_inv})
    model_data = model_data.replace({'Majority_Vote':label_dict_inv})

    
    print 'Save results to model_data.csv in ', path
    model_data.to_csv(path + kmeansoutname, sep = '\t')
#%%

if (parameter_tuning =='yes' or compare_cluster_methods == 'yes' or brute_force_feature_selection == 'yes' or cf_matrix == 'yes' or compare_unsup_data_with_ground_truth == 'yes'):
    print 'Beende Programm - Plotten confusion matrix und/oder parameter Optimierung und oder kmeans/boden vergleich aktiviert'
    sys.exit(0)  
    
    
############################################KLASSIFIZIERUNG#################### 
############################################KLASSIFIZIERUNG#################### 
############################################KLASSIFIZIERUNG#################### 
############################################KLASSIFIZIERUNG#################### 
############################################KLASSIFIZIERUNG#################### 
############################################KLASSIFIZIERUNG#################### 
#%%
# Run supervised classification
if class_data_present == 'no':
    print('Ohne Klassifizierungsdaten keine Klassifikation - hui')
    sys.exit(1)
    
if sup_unsup == 'sup':
    print 'Supervised Classification'
    # iterate over classifiers
    #make basic sanity check:
    # We use the full dataset to train because we used cross validation to get the optimal parameters, did we?      
    
    for idx, clf in enumerate(classifiers):
        print names[idx]
        clf.fit(model_X, model_y)
        pred = clf.predict(class_X)
        class_data[names[idx]] = pred

        
        calibrated_clf = CalibratedClassifierCV(clf,method = 'sigmoid')
        calibrated_clf.fit(model_X, model_y)
        probas = calibrated_clf.predict_proba(class_X)
        
        # Uncomment to wirte probabilities for each class
        #for i, l in enumerate(label_dict.keys()):
        #    class_data['Probabilities_' + l + '_' + names[idx]] = probas[:,i]
        class_data['Max Probability' + '_' + names[idx]] = np.amax(probas, axis=1)
     
        class_data = class_data.replace({names[idx]:label_dict})
        
        if make_sup_grid == 'yes':
                print "Gridding of the sup data"
                column_name = names[idx]
                griddf = class_data[['X','Y', column_name]]
                griddf.to_csv(path + 'temp.xyz', header = False, index=False)
                out_grid_name = path + out_grid_base + str(column_name) + '.grd'
                print "Gridding using GMT to ", out_grid_name
                ymin = griddf['Y'].min()
                ymax = griddf['Y'].max()
                xmin = griddf['X'].min()
                xmax = griddf['X'].max()
                os.chdir(path)
                command = 'gmt nearneighbor temp.xyz -G' + out_grid_name + ' -I' + str(grid_resolution) + ' -S' + str(grid_search_radius)  \
                            +' -V -R'+str(xmin)+'/'+str(xmax)+'/'+str(ymin)+'/'+str(ymax)
                os.system(command)    
    
    # Add majority vote
    temp = class_data[names].mode(axis=1)
    class_data['Majority_Vote'] = temp[0] 
    
    
    # Labels wiederherstellen
    class_data = class_data.replace({'Label':label_dict_inv})
    class_data = class_data.replace({'Majority_Vote':label_dict_inv})
    for i in range(len(names)):
        class_data = class_data.replace({names[i]:label_dict_inv})
        
    # Save results
    class_data.to_csv(path + out)
    

# Run unsupervised classification
if sup_unsup == 'unsup':
    for n_clusters in range_n_clusters:
        print 'Kkmeans with cluster number: ', n_clusters
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(class_X)
        column_name = 'Cluster_n_' + str(n_clusters)
        class_data[column_name] = cluster_labels

    # Mache X und Y Felder
    class_data['GEOMETRY'], class_data['X'], class_data['Y'] = class_data['WKT_GEOMETRY'].str.split(' ', 2).str
    #Get rid of ()
    class_data['X'] = class_data['X'].map(lambda x: x.lstrip('()').rstrip('()'))
    class_data['Y'] = class_data['Y'].map(lambda x: x.lstrip('()').rstrip('()'))
    class_data[["X", "Y"]] = class_data[["X", "Y"]].astype(float) 
    class_data.to_csv(path + out, sep = '\t')
    
    if make_kmeans_grid == 'yes':
        #Lösung mit gmt 
        for n_clusters in range_n_clusters:
            column_name = 'Cluster_n_' + str(n_clusters)
            griddf = class_data[['X','Y', column_name]]
            griddf.to_csv(path + 'temp.xyz', header = False, index=False)
            out_grid_name = path + out_grid_base + str(n_clusters) + '.grd'
            print "Gridding using GMT to ", out_grid_name
            ymin = griddf['Y'].min()
            ymax = griddf['Y'].max()
            xmin = griddf['X'].min()
            xmax = griddf['X'].max()
            os.chdir(path)
            command = 'gmt nearneighbor temp.xyz -G' + out_grid_name + ' -I' + str(grid_resolution) + ' -S' + str(grid_search_radius)  \
                        +' -V -R'+str(xmin)+'/'+str(xmax)+'/'+str(ymin)+'/'+str(ymax)
            os.system(command)

        