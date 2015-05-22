function [AUC,tpr,fpr,b]=svmroccurve(xtest,ytest,xsup,w,w0,kernel,kerneloption,span)

% USAGE
%
% [AUC,tpr,fpr,b]=svmroccurve(xtest,ytest,xsup,w,w0,kernel,kerneloption,span)
%
%  process the ROC curve and the AUC for SVM model (either SVMROC or SVM L2) 
% 
%  the inputs are as usual for SVM
%
% if nargin == 2 then the entries are
% 
% [AUC,tpr,fpr,WMW,b]=svmroccurve(ypred,ytest);
%
%
% the outputs
%
% AUC       Area under curve value
% tpr,fpr   true positive and false positive vector for ROC curve plotting purpose
% b         a new bias for the decision function. b can replace w0 and it has been
%           processed so that the decision function corresponds to the one where the roc
%            curve and the (1,0)-(0,1) diagonal meets
%
%

%   
% 

% 30/07/2004  A. Rakotomamonjy


eps=1e-7;
classcode=[1 -1];

if  nargin >2
    ypred = svmrocval(xtest,xsup,w,w0,kernel,kerneloption,span);
else
    ypred=xtest;  
    w0=0;
end;
npos=sum(ytest==1);
nneg=sum(ytest==-1);

%     
% %WMW statistics
% 
% indpos=find(ytest==classcode(1));
% indneg=find(ytest==classcode(2));
% A=0;
% for i=1:npos
%     A=A+ sum( (ypred(indpos(i))- ypred(indneg)) > eps);
%     
% end;
% WMW=A/npos/nneg;
% AUC=WMW;
% 
% if nargout >1
%     
%     N=length(ypred);
%     [ypred,ind]=sort(ypred);
%     ytest=ytest(ind);
%     b=[min(ypred)-1;ypred];
%     for i=1:N+1;
%         hi=sign(ypred-b(i)-eps);
%         [Conf,metric]=ConfusionMatrix(hi,ytest,classcode);
%         tp(i)=Conf(1,1);
%         fp(i)=Conf(2,1);
%     end;
% 
%     
%     tpr=tp/npos;
%     fpr=fp/nneg;
%     
%     
%     
%     %calcul de b pour un cout erreur egale à 1 pour chaque classe
%     [aux,indice]=min(abs(1-fpr-tpr)); % intersection entre la courbe roc et la diagonale.
%     b=w0-b(indice);
% end;


ytest=ytest>0;
[ypred,ind] = sort(ypred);
ytest       = ytest(ind);


fpr = cumsum(ytest)/sum(ytest);
tpr = cumsum(1-ytest)/sum(1-ytest);
tpr = [0 ; tpr ; 1];
fpr = [0 ; fpr ; 1];
n = size(tpr, 1);
AUC = sum((fpr(2:n) - fpr(1:n-1)).*(tpr(2:n)+tpr(1:n-1)))/2;
b=[min(ypred)-1;ypred];
[aux,indice]=min(abs(1-fpr-tpr)); % intersection entre la courbe roc et la diagonale.
b=w0-b(indice) + eps;