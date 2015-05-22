function [xsup,ysup,w,b,alpha,pos] = svmreg(x,y,C,epsilon,kernel,kerneloption,lambda,verbose,span,framematrix,vector,dual)
  
%  USAGE 
% [xsup,ysup,w,b,alpha,pos] = svmregL2(x,y,C,epsilon,kernel,kerneloption,lambda,verbose,span,framematrix,vector,dual)
% 
% This function process the SVM regression model using a linear epsilon insensitive cost.
% and L2 penalization of outside the tube examples.
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
%
%   OUTPUT
%
%   alpha and pos are the outputs of the dual (hence alpha is size 2*n)



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
H = Idif'*H*Idif + eye(2*size(H,1))/C;
c = [-epsilon+y ; -epsilon-y];  % [ alpha^* alpha]
A = [-ones(1,n)  +ones(1,n) ]';
b=0;   
     





Cinf=inf;
    [alpha,bias,pos]=monqp(H,c,A,b,Cinf,lambda,verbose,x,ps);
    
%    [alpha,bias,pos]=monqpCinfty(H,c,A,b,lambda,verbose,x,ps);
    

aux=zeros(length(H),1);
aux(pos)=alpha;
alpha=aux;
newpos=find(alpha(1:n)>0|alpha(n+1:2*n)> 0);
w = alpha(newpos)-alpha(n+newpos);  %[ alpha^* alpha]

xsup = x(newpos,:);
ysup = y(newpos);
nsup =length(newpos);
b=bias;

xsup = x(newpos,:);
ysup = y(newpos);
nsup =length(newpos);
b= - bias;   % do the math for checking the sign  (if you change the sign of A, you can change the sign of bias
