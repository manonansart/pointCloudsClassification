function ypred=rankboostAUCval(x,alpha,threshold,rankfeat,T);

% USAGE
%
% ypred=rankboostAUCval(x,alpha,threshold,rankfeat,T);
%
% evaluate a rankboost decision function of data x
%
%  the inputs are 
%  
%   alpha  : vector  of weigth of each weak learner
%   threshold : vector of each weak learner translation
%   rankfeat : vector of each weak learner feature.
%   T       : number of weak learners
%
% see also rankboostAUC

% 30/07/2004  A. Rakotomamonjy

ypred=zeros(size(x,1),1);
for i=1:T
    ypred=ypred+alpha(i)*(x(:,rankfeat(i))>threshold(i));
end;
