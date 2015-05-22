function y=rnval(xapp,xtest,kernel,kerneloption,c,d, T);

% USAGE
%    y=rnval(xapp,xtest,kernel,kerneloption,c,d,T);
%
%    y= K*c+ T*d   
%    calculates the output y of a Regularization networks given
%    a learning dataset xapp, points xtest  and kernel
%
% Input
%   xapp       Input Learning dataset      
%   xtest      Iput  Test     dataset
%   kernel     The Kernel
%
%
%   kerneloption : parameters of kernel
%   
%   c,d         coefficients of the RN obtained from rncalc fucntion
%   T           span matrix  Tij= phi_j(x_i) (defaults = 1) 
%    
% Output
%
%  y    y=K*c+T*d


%  11/03/2002 Alain Rakotomamonjy
 
nt=size(xtest,1);
if nargin <7
    T=ones(nt,1);
else
    if isempty(T)
            T=zeros(nt,1);
        end;
    if size(T,1)~=nt
        error('Span matrix T and xtest must have the same number of rows...');
    end;
    
end;
if nargin <6
    error('Not enough input parameters...');
end;
n=size(xapp,1);
K  =  zeros(n,n);		
K=svmkernel(xtest,kernel,kerneloption,xapp);
y=K*c+T*d;