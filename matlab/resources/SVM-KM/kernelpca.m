function [eigvect,eigval,Kt]=kernelpca(x,kernel,kerneloption)

% USAGE
%   [eigvect,eigval]=kernelpca(x,kernel,kerneloption)
%
% Diagonalizing the covariance matrix in feature space 
%
% eigenvalues are sorted in descending order
%
%

tol=1e-15; %tolerance on zeroness of eigenvalue
if nargin <2
    kernel='poly';
    kerneloption=1;
end;

nx=size(x,1);
K=svmkernel(x,kernel,kerneloption);
oneM=ones(nx,nx)./nx;
Kt=K-oneM*K-K*oneM+oneM*K*oneM;
[eigvect,eigval]=eig(Kt/nx); % diagonalizing Kt/nx;
eigval=diag(eigval);
indeigval=(find(abs(eigval)>=tol)); % keeping only eigval higher than tol
eigvect=eigvect(:,indeigval);
eigval=eigval(indeigval);
nbeigval=length(eigval);
for i=1:nbeigval % normalizing eigenvector
    eigvect(:,i)=eigvect(:,i)/sqrt(eigval(i));
end;

[aux,ind]=sort(-eigval);
eigval=eigval(ind);
eigvect=eigvect(:,ind);