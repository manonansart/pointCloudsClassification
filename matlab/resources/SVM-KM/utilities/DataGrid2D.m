function  [xtest,xtest1,xtest2,nn1,nn2]=DataGrid2D(vectX,vectY);

% [xtest,xtest1,xtest2,nn]=DataGrid2D(vectX,vectY);

[xtest1 xtest2]  = meshgrid(vectX,vectY); 
[nn1,nn2]= size(xtest1); 
xtest = [reshape(xtest1 ,nn1*nn2,1) reshape(xtest2 ,nn1*nn2,1)]; 
