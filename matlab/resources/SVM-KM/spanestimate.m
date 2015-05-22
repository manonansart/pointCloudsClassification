function loo = spanestimate(x,y,kernel,kerneloption,alpha,pos,b,C);

% USAGE
%
% loo = spanestimate(x,y,kernel,kerneloption,alpha,pos,b,C);
%
% gives an estimate of the LOO error of a soft margin SVM
% 
% x and y               are the training examples and their labels
% kernel, kerneloption  define the kernel and its parameters (this is used only for
%                       processing the Gram Matrix of the training data
% alpha,pos             define the lagrangian multiplier associated to the SVM
% b                     the bias of the SVM
% C                     penalization coefficient.

if ~strcmp(kernel,'numerical')
    K=svmkernel(x,kernel,kerneloption);
else
    K=kerneloption.matrix;
end
alpha1=zeros(size(y));
alpha1(pos)=alpha;
output= y.*(K*(alpha1.*y)+b);

eps=1e-5;
sv1=find(alpha1>max(alpha1)*eps & alpha1 < C*(1-eps));
sv2=find(alpha1 > C*(1-eps));


if isempty(sv1)
    loo=mean(output<0);
    return
end;


ell=length(sv1);
KSV=[K(sv1,sv1) ones(ell,1);[ones(1,ell) 0]];
invKSV=inv(KSV+diag(1e-12*[ones(1,ell) 0]));

n=length(K);
span=zeros(n,1);
tmp=diag(invKSV);

span(sv1)=1./tmp(1:ell);

if ~isempty(sv2);
    V=[K(sv1,sv2); ones(1,length(sv2))];
    span(sv2)=diag(K(sv2,sv2))-diag(V'*invKSV*V);
end;

loo=mean(output-alpha1.*span < 0);

