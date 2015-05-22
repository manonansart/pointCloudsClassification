function [ypred,tabkppv,distance]=kernelknn(xapp,yapp,valY,kernel,kerneloption,X,k)

%
% knn implementation
%
% USE : [ypred,tabkppv,distance]=kernelknn(xapp,yapp,valY,kernel,kerneloption,X,k)
%
% xapp, yapp : learning data
% valY : all the Y value possible
% kernel : the inner product in the feature space
% kerneloption : option of the kernel (gaussian bandwidth or degree of polynomial kernel)
% X : data to be classified
% k : number of nearest neighbours
%
% ypred : class of X
% tabkppv : [nbpts x nbpts] index of nearest neighbours
% distance : [nbpts x nbpts] distance
%
% Vincent Guigue 08/01/03

% check nargin

if nargin<7
  error('too few argumemnts');
elseif nargin<7
  k=3;
else
  if mod(k,2)==0
    error('k must be odd');
  end
end

if size(xapp,2)~=size(X,2)
  error('dimension incompatibility');
end


ndim = size(xapp,2);
nptxapp = size(xapp,1);
nptX = size(X,1);



kxx=diag(svmkernel(xapp,kernel,kerneloption,xapp));
kyy=diag(svmkernel(X,kernel,kerneloption,X));
kxy=svmkernel(xapp,kernel,kerneloption,X);
distance=repmat(kxx,1,nptX)+ repmat(kyy',nptxapp,1)-2*kxy;








[val kppv] = sort(distance,1);

% bilan sur les k premieres lignes
kppv = reshape(kppv(1:k,:),k*nbtest,1);
Ykppv = yapp(kppv,1);
Ykppv = reshape(Ykppv,k,nbtest);

% trouver le plus de reponses identique par colonne

tabkppv = Ykppv;

vote = [];
for i=1:nbtest
  for j=1:length(valY)
    vote(j,i)=size(find(Ykppv(:,i)==valY(j)),1);
  end
end

[val ind]=max(vote,[],1);
ypred = valY(ind);