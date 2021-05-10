# pointCloudsClassification

This Matlab code implements solutions from the paper <a href="http://www.cs.jhu.edu/~misha/Papers/Johnson99.pdf">"Using Spin Images for Efficient Object Recognition in Cluttered 3D Scenes"</a>, by Andrew E. Johnson and Martial Hebert. The goal is to identify 3D point clouds representing an object on the road (pedestrian, car...).

The choice of attributes was based on the recommendations of the article. We used statistics on the intensity of the points, the bounding box and the attributes scatter-ness, linear-ness and surface-ness. As one class was over-represented compared to the others, we also rebalanced the classes.

The 'dish_area' folder contains code for binary classification, applied on the dish area dataset. This code compares a gaussian SVM, linear SVM and k-means algorithm.

The 'lomita' folder contains code for multi-class classification. We used a linear SVM and compared one versus all and one versus one strategies. For the one versus one SVM, 2 hyper-parameter tuning strategies were used : in the first one, the hyperparameters was chosen for each SVM one versus one; in the second one ("simplified") the same hyperparameter value was used for all SVMs

This code was created for a school project.
