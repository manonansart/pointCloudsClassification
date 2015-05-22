

function  [beta,r2,pos]= r2smallestsphere(xapp,kernel,kerneloption,Cpenalization,betaall)

% usage
% [beta,r2,pos]= r2smallestsphere(xapp,kernel,kerneloption,Cpenalization,betaall)
%------------------------------------------------------------------------------------
%
%       Processing the radius of the smallest sphere including learning data
%
%-------------------------------------------------------------------------------------
%
%
% if you are using a linear slacks SVM then Cpenalization is equal to inf
% otherwise as you are changing the kernel you have to take C into account
% and then K=K+ 1/Cpenalization*I
%
% if you want to pass just the kernel matrix, then pass the struct with a field
% 'matrix'

if nargin < 6
    betaall=[];
end;
if nargin < 5
    Cpenalization=inf;
end;

if ~isempty(xapp)
    
    [kapp]=svmkernel(xapp,kernel,kerneloption);
    if Cpenalization~=0
        kapp=kapp+(1/Cpenalization)*eye(size(kapp));
    end;
elseif isfield(kerneloption,'matrix')
    kapp=kerneloption.matrix+(1/Cpenalization)*eye(size(kerneloption.matrix));
else
    error('No ways for processing the radius. Check inputs...');
end;

D=diag(kapp);
A = ones(size(D));
b=1;
lambda=1e-7;
verbose=0;
C=inf;

[betaaux,lagrangian,pos]=monqpCinfty(2*kapp,D,A,b,lambda,verbose,[],[],betaall); 
beta=zeros(size(D));
beta(pos)=betaaux;
r2=-betaaux'*kapp(pos,pos)*betaaux + betaaux'*D(pos);
