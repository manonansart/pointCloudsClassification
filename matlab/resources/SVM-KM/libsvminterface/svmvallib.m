function   ypred=svmvallib(x,xsup,w,b,kernel,kerneloption,span)
    
% USAGE 
%
% ypred=svmvallib(x,xsup,w,b,kernel,kerneloption,span)
%
    
switch kernel
    case  'gaussian'
        kerneltype=2;
    case   'poly'
        kerneltype=1;
    otherwise
        kerneltype=0;
        degree=1;
 end;
coefficient=0;
gamma=0;
if kerneltype==2
    degree=0;
    gamma= 1/2/kerneloption.^2;
end;
if kerneltype==1 & kerneloption==1;
    coefficient=0;
    degree=1;
    gamma=1;
end;
if kerneltype==1 & kerneloption>1;
    coefficient=1;
    degree=kerneloption;
    gamma=1;
end;
Cache=40;
C=0;
epsilon=1e-7;
svmtype=0;
parameters=[kerneltype degree gamma coefficient C Cache epsilon svmtype];
  
 ypred=ones(1,size(x,1));
[ClassRate, ypred]= mexSVMClass(x', ypred, w', xsup',-b,parameters);
ypred=ypred';