function [xsup,alpha,multiplier,pos,Ksup]=svmoneclass(x,kernel,kerneloption,nu,verbose,alphainit)

% USAGE
%
% [xsup,alpha,rho,pos]=svmoneclass(x,kernel,kerneloption,nu,verbose)
%
% model f(x) = <w,x> - rho 
%
% problem solved 
% 
% min 1/2 <w,w>  + 1/nu/ell \sum_i \xi_i - rho
% st  <w,xi> - rho >= xi_i 
%
% the output is rho is actually - rho

if ~isempty(x)
[n1,n2]=size(x);
K=normalizekernel(x,kernel,kerneloption);
else
   
    K=kerneloption.matrix;
    n1=size(K,1);
end;

%K=svmkernel(x,kernel,kerneloption);
c=zeros(n1,1);
A=ones(n1,1);
b=1;
C= 1/nu/n1;
lambda=1e-8;
if nargin <6
    alphainit=C/2*ones(n1,1);
    alphainit=[];
else
    alphainit(find(alphainit>=C))=C;
end;
[alpha, multiplier, pos]=monqp(K,c,A,b,C,lambda,verbose,[],[],alphainit);
if ~isempty(x)
xsup=x(pos,:);
else
    xsup=[];
end;
Ksup=K(pos,pos);