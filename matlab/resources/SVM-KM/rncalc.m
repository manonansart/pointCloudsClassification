function [c,d]=rncalc(xapp,yapp,kernel,kerneloption,lambda,T)


% USAGE
%
%    [c,d]=rncalc(xapp,app,kernel,kerneloption,lambda,T);
%
%   
%    y= K*c+ T*d   
%    calculates the minimizer of equation 
%           (yapp- y)^2 + \lambda ||y||^2
%    with y= K*c+ T*d   
%
% Inputs
%    xapp       Input  Learning dataset      
%    yap        Output Learning dataset
%    kernel     The Kernel
%
%
%    kerneloption : parameters of kernel
%   
%   Lambda      regularization hyperparameters
%   T           span matrix  Tij= phi_j(x_i) (defaults = 1) 
% Outputs
%
%  c,d     so that  (yapp- K*c-T*d)^2 are minimized


%  11/03/2002 Alain Rakotomamonjy


n=size(xapp,1);
if nargin <6
    T=ones(n,1);
else
    if size(T,1)~=n & ~isempty(T)
        error('Span matrix T and xapp must have the same number of rows...');
    end;
    
end;
if nargin <5
    error('Not enough input parameters...');
end;


K  =  zeros(n,n);		K=svmkernel(xapp,kernel,kerneloption);
[c,d]=regsolve(K,T,yapp,lambda);
