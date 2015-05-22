function [xapp,yapp,xtest,ytest]=nfcvreg(x,y,N,k)

% USAGE
% [xapp,yapp,xtest,ytest]=nfcvnoprior(x,y,N,k)
% N-fold crossvalidation
% k indicates the portion to leave out for test
%
nbdata=length(y);
Nportion=round(nbdata/N);

if k~=N
indtest=Nportion*(k-1)+1:Nportion*k;
else
    indtest=Nportion*(k-1)+1:nbdata;
end;
indapp=setxor(1:nbdata,indtest);
xapp=x(indapp,:);
xtest=x(indtest,:);
yapp=y(indapp);
ytest=y(indtest);

