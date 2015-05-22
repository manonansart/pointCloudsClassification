function   [f,f1,f2]= functioneval(x,fonction,a,b)

% Usage f= ferreur(x,fonction,a,b)
% 
% Evaluation of some useful functions
% for kernel hyperparameters tuning.
%
% 'heaviside'
% 'expon'     1./(1+exp(-a*x+b))  
% 'dexpon'    expon derivative
%
%
% 07/08/2000 AR

thresh=40;
switch lower(fonction)
case 'heaviside'
   f=x>0;
   
% x is thresholded to avoid Matlab or Windows crashing

case 'expon'
   x=x.* (abs(x)<thresh) + thresh.* (x>=thresh)-thresh.* (x<=-thresh); 
   f=1./(1+exp(-a*x+b));
   
case 'dexpon'
    x=x.* (abs(x)<thresh) + thresh.* (x>=thresh)-thresh.* (x<=-thresh); 

   f= a.*exp(-a*x+b)./(1+exp(-a*x+b)).^2;
end;