function [distance]=calcdistance(x1,x2)

%
% fonction de calcul de la distance 
% notament utiles pour knn
%

if nargin<2
  x2=x1;
end

ndim = size(x1,2);
nptx1 = size(x1,1);
nptx2 = size(x2,1);

% distance de X a xapp :
mat1 =  repmat(x1, nptx2,1);

mat22 = repmat(x2,1,nptx1)';
mat2 = reshape(mat22 ,ndim, nptx1*nptx2)';
distance = mat1 - mat2 ;

distance = sum(abs(distance),2);
distance = reshape(distance,nptx1,nptx2);