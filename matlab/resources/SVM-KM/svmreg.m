function [xsup,ysup,w,b,newpos,alpha,obj] = svmreg(x,y,C,epsilon,kernel,kerneloption,lambda,verbose,span,framematrix,vector,dual,alphainit)
  
%  USAGE 
% [xsup,ysup,w,b] = svmreg(x,y,C,epsilon,kernel,kerneloption,lambda,verbose,span,framematrix,vector,dual)
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


if nargin < 13
    alphainit=[];
end;

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


n = length(y);
ps  =  zeros(n,n);		
ps=svmkernel(x,kernel,kerneloption,[],framematrix,vector,dual);
H = ps; 
I = eye(n);
Idif = [I -I];
H = Idif'*H*Idif;
c = [-epsilon+y ; -epsilon-y];

if ~isempty(span)
    fprintf('semiparametric SVM\n');
    A=[span' -span']';
    [nbpoint,nbcontrainte]=size(A);
    size(H);
    b=zeros(nbcontrainte,1);
else
    A = [ones(1,n)  -ones(1,n) ]';
    b=0;   
end; 





if C~=inf
    [alpha,bias,pos]=monqp(H,c,A,b,C,lambda,verbose,x,ps,alphainit);
else
    [alpha,bias,pos]=monqpCinfty(H,c,A,b,lambda,verbose,x,ps);
end;

aux=zeros(length(H),1);
aux(pos)=alpha;
alpha=aux;
newpos=find(alpha(1:n)>0|alpha(n+1:2*n)> 0);
w = alpha(newpos)-alpha(n+newpos); 
if ~isempty(x)
    xsup = x(newpos,:);
else
    xsup=[];
end
ysup = y(newpos);
nsup =length(newpos);
b=bias;
obj=-0.5*alpha'*H*alpha + c'*alpha;
