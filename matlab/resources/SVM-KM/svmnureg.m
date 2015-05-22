function [xsup,ysup,w,b,alpha,epsilon] = svmreg(x,y,C,nu,kernel,kerneloption,lambda,verbose)
  
%  USAGE 
% [xsup,ysup,w,b] = svmreg(x,y,C,nu,kernel,kerneloption,lambda,verbose,span,framematrix,vector,dual)
% 
% This function process the SVM regression model using a linear epsilon insensitive cost.
%
% 
% 
% INPUT
%
% Training set
%    x  : input data 
%    y  : output data
% parameters
%		C		: Bound on the lagrangian multipliers     
% 		epsilon 	 : e-tube around the solution
%		kernel		: kernel function. classical kernels are
%
%		Name			parameters
%		'poly'		polynomial degree
%		'gaussian'	gaussian standard deviation
% 
%			for more details see svmkernel
%		
% 		kerneloption : parameters of kernel
%
%
%   lambda : conditioning parameter for QP methods
%   verbose : display outputs (default value is 0: no display)
%
%	 span     : Matrix for semiparametric learning (default span=[])	
%
%   ----- 1D Frame Kernel -------------------------- 
%
%   framematrix  frame elements for frame kernel
%   vector       sampling position of frame elements
%	 dual 		  dual frame
%
%   see also svmclass,svmreg, svmval
%
%	

%	21/09/97 S. Canu
%   06/06/00 A.Rakotomamonjy Including SVMkernel


if nargin < 3
    C = 1000;
end;

if nargin < 4
    epsilon = 0.1;
end;

if nargin < 5
    kernel='gaussian';
    kerneloption = 1;
end;

if nargin < 7
    lambda = .0000001;
end;
if nargin <8
    verbose=0;
end;
if nargin <9
    span=[];
    semiparam=0;
end;
if nargin < 10
    framematrix=[];
    vector=[];
    dual=[];
end;


n = size(x,1);
ps  =  zeros(n,n);		
ps=svmkernel(x,kernel,kerneloption,[],framematrix,vector,dual);
H = ps; 
I = eye(n);
Idif = [I -I];
H = Idif'*H*Idif;
c = [-y/C ; +y/C];


 A1 = [ones(1,n)  -ones(1,n) ]';
 A2 = [ones(1,n)  ones(1,n) ]';
 A=[A1 A2];
 b=[0;nu];   



%keyboard

Up=(1/n)*ones(2*n,1);
if C~=inf
    [alpha,lagrangmul,pos]=monqp(H,c,A,b,Up,lambda,verbose,x,ps);
else
    [alpha,bias,pos]=monqpCinfty(H,c,A,b,lambda,verbose,x,ps);
end;

aux=zeros(length(H),1);
aux(pos)=alpha;
alpha=aux;
newpos=find(alpha(1:n)>0|alpha(n+1:2*n)> 0);
w = (-alpha(newpos)+alpha(n+newpos))*C; 

xsup = x(newpos,:);
ysup = y(newpos);
nsup =length(newpos);


b=-lagrangmul(1)*C;
epsilon=-lagrangmul(2)*C;
