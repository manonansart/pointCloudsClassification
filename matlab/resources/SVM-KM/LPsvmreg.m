function    [xsup,ysup,w,b,pos]=LPsvmreg(xapp,yapp,C,epsilon,lambda,kernel,kerneloption,verbose)

%  USAGE
%
%   [xsup,w,b,pos]=LPsvmclass(xapp,yapp,C,epsilon,lambda,kernel,kerneloption,verbose)
%
%   Linear Programming SVM  (see Mangasarian 1998)
%   
%   f(x)= sum_ w_j K(x,x_j) +b 
%
%  this function returns xsup, w and b
%



Napp=size(xapp,1);

% 
%U=zeros(Napp,Napp+Napp+Napp+1,1); % u, v,eta, nu et b  In the non linear case w has the same length as x
f=zeros(Napp+Napp+Napp+Napp+1,1);
f(1:2*Napp)=1;            %  penalizing u and v
f(2*Napp+1:4*Napp)=C;    %  penalizing eta and nu

K=svmkernel(xapp,kernel,kerneloption);

A= [-K K  -eye(Napp) zeros(Napp,Napp) -1*ones(Napp,1)];

A=[A; K -K  zeros(Napp,Napp) -eye(Napp)  1*ones(Napp,1)];  

b=epsilon*ones(2*Napp,1) + [-yapp;yapp];
lb=[zeros(4*Napp,1);-inf];
tic
x=linprog(f,A,b,[],[],lb,[]);

w=x(1:Napp,1)- x(Napp+1 : 2*Napp,1) ;
pos=find(abs(w) > 1e-6);
xsup=xapp(pos,:);
w=w(pos);
b=x(end);  
ysup=yapp(pos);