function  feature=kernelpcaproj(xpca,xtest,eigvect,kernel,kerneloption,eigvectind);

%
%
%
%
%
%
nxpca=size(xpca,1);
nxtest=size(xtest,1);
Ktest=svmkernel(xtest,kernel,kerneloption,xpca); 
K=svmkernel(xpca,kernel,kerneloption); 
% centering in features spaces
onepca=ones(nxpca,nxpca)/nxpca;
onetest=ones(nxtest,nxpca)/nxpca;
Kt=Ktest-onetest*K-Ktest*onepca+onetest*K*onepca;
% projection on eigenvector

feature=Kt*eigvect(:,eigvectind);
