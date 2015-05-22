function [K,meanK,stdK]=normalizekernelLAR(K,meanK,stdK,solution)

% USAGE 
%
% [K,meanK,stdK]=normalizekernelLAR(K,meanK,stdK,solution)
%
% treat K as an information source and normalize K.
% 
% the three last inputs are  already preprocessed normalization parameters.


n=size(K,1);
if nargin<2;
    meanK=mean(K);
    stdK= std(K);
    K= (K-ones(n,1)*meanK)./(ones(n,1)*stdK);
else
    indxsup=solution.indxsup;
     K= (K-ones(n,1)*meanK(indxsup))./(ones(n,1)*stdK(indxsup));
    
end;

