function [xsup,w,b]=svmclasslib(x,y,C,lambda,kernel,kerneloption,verbose,span)


% USAGE
%
% [xsup,w,b]=svmclasslib(x,y,C,lambda,kernel,kerneloption)
%
% simple interface to SVM LIB code for C-SVM


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
epsilon=1e-7;
svmtype=0;
parameters=[kerneltype degree gamma coefficient C Cache epsilon svmtype];


  %  [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = mexSVMTrain(Samples, Labels, Parameters, Weight);
  
  [w, xsup, b ] = mexSVMTrain(x',y', parameters);
  
  w=w';
  xsup=xsup';
  b=-b;  % This is for compatibility with SVM-KM toolbox 
    
  % This is for compatibility with SVM-KM toolbox 
  % This is because it seems that the 1rst need to be positive
    if y(1)==-1 
        w=-w;
        b=-b;
    end;
    
    