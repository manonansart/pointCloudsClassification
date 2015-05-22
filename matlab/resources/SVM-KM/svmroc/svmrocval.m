function ypred=svmrocval(xtest,xsup,w,w0,kernel,kerneloption,span)

% USAGE
%
%  ypred=svmrocval(xtest,xsup,w,w0,kernel,kerneloption,span)
%  
%  process the output of an SVM model that optimize roc curve
%  this code works for SVM-ROC or SVML2 code 
% 

%  30/07/2004 A. Rakotomamonjy


if iscell(xsup)
    [ypred]=svmval(xtest,xsup{1},w,w0,kernel,kerneloption,span)-svmval(xtest,xsup{2},w,w0,kernel,kerneloption,span)+w0;
else
   
    [ypred]=svmval(xtest,xsup,w,w0,kernel,kerneloption,span);
end;