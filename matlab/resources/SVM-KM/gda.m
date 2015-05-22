function [Xt,Xapp,Alpha,kM,valBeta,wM]=gda(xi,yi,X,nbclass,kernel,kerneloption,ndimfin)


%   USAGE
%     [Xt,Xapp,Alpha,kM]=gda(xi,yi,X,nbclass,kernel,kerneloption,ndimfin)
%
%  This function is a script that process the
%  Linear Discriminant Analysis .
%
%   xi,yi : learning data
%   Xt : testing data
%   nbclass : number of class
%   ndimfin (def=nbclass-1): output dimention
%
%   xp : projected learn data
%   Xtp : projected test data  
%   x1p,x2p,x3p : projected learn data by class
%
%   P : discrimant axis
%   VP : associated eigen value.

% Gaston Baudat & Fatiha Anouar / 21st October 2000 / Exton PA
% 19341 USA
% Vincent Guigue / 5th march 2003
 
if nargin<7
  ndimfin = nbclass-1;
end

% tri des points par classe -> matrice de poids !
xi_tri = [];
yi_tri = [];
ordre_mem = [];
for i=1:nbclass
  index = find(yi==i);
  xi_tri = [xi_tri; xi(index,:)];
  yi_tri = [yi_tri, i*ones(length(index),1)];
  ordre_mem = [ordre_mem; index];
end

xi = xi_tri;
yi = yi_tri;


n = nbclass; % nb of class
nbpts = size(xi,1); % nb pts
nbpts_t = size(X,1); % nb pts

for i=1:nbclass
  nbpts_i(i,1)=length(find(yi==i));
end


% kernel
kM = svmkernel(xi,kernel,kerneloption);
kMt= svmkernel(X,kernel,kerneloption,xi);
%keyboard

% Uncentred -> centred
kM = kM - (1/nbpts*(ones(nbpts)*kM)) - (1/nbpts*(kM'*ones(nbpts))) + 1/(nbpts^2)*(ones(nbpts)*kM*ones(nbpts));
%keyboard

kMt = kMt - (1/nbpts_t*(ones(nbpts_t)*kMt)) - (1/nbpts*(kMt*ones(nbpts))) + 1/(nbpts_t*nbpts)*(ones(nbpts_t)*kMt*ones(nbpts));

[veckM,valkM]=eig(kM);

% sort valkM
aux = diag(valkM);
[val,ind_sort] = sort(-aux);
valkM = diag(aux(ind_sort));
veckM = veckM(:,ind_sort);

% Rank -> new rank
minVal=max(diag(valkM))/1000; 
elimin = find(diag(valkM) < minVal);
rankkM = nbpts-length(elimin);

% modify kM
keep = setdiff([1:nbpts],elimin);
valkM=valkM(keep,keep);
veckM=veckM(:,keep);


kM=veckM*valkM*veckM';
fprintf('Estimated rank of K: %d (elimin %d var)\n',rankkM,length(xi)-rankkM);


% weights matrix
ptr=1;
for i=1:nbclass
  index = ptr:ptr+nbpts_i(i,1)-1;
  %index = find(yi==i); % les points sont tries !
  
  wM(index,index)=1/nbpts_i(i,1)*ones(nbpts_i(i,1));
  
  ptr = ptr+nbpts_i(i,1);
end

%compute alpha normalized vectors
nbAxes=min([rankkM ; nbpts-1 ; ndimfin]);
[vecBeta,valBeta]=eig(veckM'*wM*veckM); % ???

% recuperer les nbAxes + grande
% eliminer les complexes

aux = diag(valBeta);
ind_noimag = find(imag(aux)==0 & real(aux)>10e-6);
[val,ind] = sort(-aux(ind_noimag));
ind_sort = ind_noimag(ind);

valBeta = diag(aux(ind_sort));
vecBeta = vecBeta(:,ind_sort); 

tmp = veckM*inv(valkM)*vecBeta;

for i=1:nbAxes
  Alpha(:,i)=tmp(:,i)/sqrt(tmp(:,i)'*kM*tmp(:,i));
end


Xapp = kM*Alpha(:,1:nbAxes);
% remise dans l'ordre...
newordre(ordre_mem) = [1:nbpts];
Xapp = Xapp(newordre,:);
Xt = kMt*Alpha(:,1:nbAxes);

if newordre~=[1:nbpts];
  fprintf('Changement de l''ordre des points...\n');
end


