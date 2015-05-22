%   This is a set of m. files that provides functions
%   for AUC optimization by SVM and rankboost. most of these files
%   have been used for producing the results in the SVM and ROC curve
%   Tech Rep.
%  
%
%   30/07/2004 A. Rakotomamonjy


%----------------------- Code------------------------------
% svmroc                    SVM-ROC training algorithm
%
% svmroccurve               SVM-ROC roc curve and AUC evaluation
%
% svmrocval                 SVM-ROC testing algorithm
%
% svmrocCS                  SVM-ROC training algorithm for small datasets (no decomposition)
%
% rankboostAUC              rankboost training algorithm for dipartite data
%
% rankboostAUCval           rankboost testing algorithm
%
% rankroccurve              rankboost roc curve and AUC evaluation
%
%--------------------- Examples ----------------------------
%
%
% exsvmroc                  example of SVM-ROC on real-data
%
% exsvmroc1                 example of SVM-ROC on toy data with comparison with
%                           SVM decision function
%
% exsvmroc2                 cross-validation code for SVM-ROC on real data
%
% exsvmrocCsigma            code for analysing AUC performance of  SVM-ROC with regards to
%                           data skewness and kernel parameters.
%
% exsvmroctoy               code for analysing AUC performance of  SVM-ROC with regards to
%                           margin and neighborhoods size parameters
%
% exroccurve                example of 2-norm SVM used for AUC purpose
%
% exroccurve1               cross-validation code for 2-norm SVM on real data
%
% exroccurve2               code for analysing AUC performance of 2 norm SVM with regards to
%                           data skewness and kernel parameters.
%
% exrocurvemodelsel         code for cross-validation and performance assessment wrt different
%                           parameters such as AUC, accuracy, precision or Fmeasure
%
% exrankboost               example of rankboost algorithm on real data
%
% exrankboost1              cross-validation code for rankboost on real-data