function [xapp,yapp,xtest,ytest,indice]=nfcv(x,y,N,k,classcode)

% USAGE
% [xapp,yapp,xtest,ytest]=nfcv(x,y,N,k)
% this is for classification with output code as -1 1
% so that the prior prob of each class is respected in the cv
% N-fold crossvalidation
% k indicates the portion to leave out for test
%
if nargin < 5
    classcode=[1 -1];
end;
nbdata=length(y);
Nportion=round(nbdata/N);

% indtest=Nportion*(k-1)+1:min(Nportion*k,nbdata);
% indapp=setxor(1:nbdata,indtest);
% xapp=x(indapp,:);
% xtest=x(indtest,:);
% yapp=y(indapp);
% ytest=y(indtest);

indneg=find(y==classcode(2));
indpos=find(y==classcode(1));
Nneg=length(indneg);
Npos=length(indpos);
NportionNeg=round(Nneg/N);
if k~=N
indtestNeg=indneg(NportionNeg*(k-1)+1:min(NportionNeg*k,Nneg));
else
    indtestNeg=indneg(NportionNeg*(k-1)+1:end);
end;
NportionPos=floor(Npos/N);
if k~=N
indtestPos=indpos(NportionPos*(k-1)+1:min(NportionPos*k,Npos));
else
    indtestPos=indpos(NportionPos*(k-1)+1:end);
end;
indappNeg=setxor(indneg,indtestNeg);
indappPos=setxor(indpos,indtestPos);

indtest=[indtestPos;indtestNeg];
indapp=[indappPos;indappNeg];
xapp=x(indapp,:);
xtest=x(indtest,:);
yapp=y(indapp);
ytest=y(indtest);
indice.app=indapp;
indice.test=indtest;