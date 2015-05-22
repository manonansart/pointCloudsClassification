function  [ypred,y]=backfittingval(kernelmatrix,C)

% USAGE
%
% ypred=backfittingval(kernelmatrix,C)
%

[n,n1,nbspace]=size(kernelmatrix);
%keyboard
y=zeros(n,nbspace);
y(:,1)=kernelmatrix(:,:,1)*C(:,1);
ypred=y(:,1);
for i=2:nbspace
    y(:,i)=kernelmatrix(:,:,i)*C(:,i);
    ypred=ypred+ y(:,i);
end;
