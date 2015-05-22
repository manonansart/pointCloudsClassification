function [RankedVariables,values]=FeatSelreglinearL1(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)

% Usage
%  
%  [RankedVariables,values]=FeatSelregr2w2(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)
%
%  rank variables according to their values on r2w2 when removed
%  
%
%   x,y     : input data
%   c       : penalization of misclassified examples
%   kernel  : kernel type
%   kerneloption : kernel hyperparameters
%   verbose
%   FeatSeloption : structure containing FeatSeloption parameters
%           Fields           
%
%           AlphaApprox : O for retraining 1, for approximation 
%           RemoveChunks : number of variable to remove (a number or 'half')
%           StopChunks   : remove 1 variable at a time when number of variables reaches this value
%
%
%

% alain.rakoto@insa-rouen.fr
%   
%
% 26/04/2004 AR
% 
%wsum=zeros(size(x,1),1);
%for T=1:20






option= optimset('Display','Off');
[Napp,Ndim]=size(x);

% 
%U=zeros(Ndim+Ndim+Napp+Napp+1,1); % u, v,eta, nu et b  In the non linear case w has the same length as x
f=zeros(Ndim+Ndim+Napp+Napp+1,1);
f(1:2*Ndim)=1;            %  penalizing u and v
f(2*Ndim+1:2*Ndim+2*Napp)=c;    %  penalizing eta and nu

A= [-x x  -eye(Napp) zeros(Napp,Napp) -1*ones(Napp,1)];
A=[A; x -x  zeros(Napp,Napp) -eye(Napp)  1*ones(Napp,1)];  
b=epsilon*ones(2*Napp,1) + [-y;y];
lb=[zeros(2*Ndim+ 2*Napp,1);-inf];
tic
x=linprog(f,A,b,[],[],lb,[],[],option);
w=(x(1:Ndim,1)- x(Ndim+1 : 2*Ndim,1))' ;

[ind,RankedVariables]=(sort(abs(w)));
values=w(RankedVariables);
RankedVariables=fliplr(RankedVariables);
values=fliplr(values);

