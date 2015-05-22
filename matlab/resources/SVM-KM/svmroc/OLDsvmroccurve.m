function [AUC,tpr,fpr,b,WMW]=svmroccurve(xtest,ytest,xsup,w,w0,kernel,kerneloption,span)

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
ypredsorted=sort(ypred);
npos=length(find(ytest==classcode(1)));
nneg=length(find(ytest==classcode(2)));
n=npos+nneg;  
i=1;
b=ypredsorted(1);
yaux=ypred-b+eps;
C=ConfusionMatrix(sign(yaux),ytest,classcode);
tp(i)=C(1,1);
fp(i)=C(2,1);

for i=1:n
    b=ypredsorted(i);
    yaux=ypred-b;
    C=ConfusionMatrix(sign(yaux),ytest,classcode);
    tp(i+1)=C(1,1);
    fp(i+1)=C(2,1);
end;
yaux=ypred-b-eps;
C=ConfusionMatrix(sign(yaux),ytest,classcode);
tp(i+2)=C(1,1);
fp(i+2)=C(2,1);
tpr=tp/npos;
fpr=fp/nneg;
difffpr=-diff(fpr);
AUC=sum(tpr(1:end-1).*difffpr);


%WMW statistics

indpos=find(ytest==classcode(1));
indneg=find(ytest==classcode(2));
A=0;
for i=1:npos
    A=A+ sum( (ypred(indpos(i))- ypred(indneg)) > eps);
    
end;
WMW=A/npos/nneg;
AUC=WMW;

% calcul de b pour un cout erreur egale à 1 pour chaque classe
[aux,indice]=min(abs(1-fpr-tpr)); % intersection entre la courbe roc et la diagonale.
if indice ==1
    b=w0-ypredsorted(indice);
else
    b=w0-(ypredsorted(indice)+ypredsorted(indice-1))/2;
end;