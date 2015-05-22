function    [xsup,w,b,pos]=LPsvmclass(xapp,yapp,C,lambda,kernel,kerneloption,verbose)

%  USAGE
%
%   [xsup,w,b,pos]=LPsvmclass(xapp,yapp,C,lambda,kernel,kerneloption,verbose)
%
%   Linear Programming SVM  (see Mangasarian 1998)
%   
%   f(x)= sum_ w_j K(x,x_j) +b 
%
%  this function returns xsup, w and b
%



Napp=size(xapp,1);


U=zeros(Napp+Napp+Napp+1,1); % w,s,eta et b  In the non linear case w has the same length as x
f=zeros(Napp+Napp+Napp+1,1);
f(Napp+1:2*Napp)=1;            %  penalizing s
f(2*Napp+1:2*Napp+Napp)=C;    %  penalizing nu

K=svmkernel(xapp,kernel,kerneloption);

A= -[K.*(yapp*yapp') zeros(Napp,Napp) eye(Napp) -yapp];

A=[A; eye(Napp,Napp) -eye(Napp,Napp) zeros(Napp,Napp) zeros(Napp,1)];  
A=[A; -eye(Napp,Napp) -eye(Napp,Napp) zeros(Napp,Napp) zeros(Napp,1)];  

b=[-ones(Napp,1); zeros(2*Napp,1)];
lb=[-inf*ones(2*Napp,1);zeros(Napp,1);-inf];
tic
if exist('lp_solve')
    sens=-ones(size(A,1),1);
    [obj,x]=lp_solve(-f,A,b,sens,lb,[]);
    if isempty(x)
        x=ones(Napp,1);
    end;
elseif exist('linprog')
    x=linprog(f,A,b,[],[],lb,[]);
else
    error('No available LP optimizer...');
end;
w=x(1:Napp,1);
pos=find(abs(w) > 1e-6);
xsup=xapp(pos,:);
w=w(pos).*yapp(pos);
b=-x(end);   % The minus is is for compatibility with svmval function and in not 
             % in agreement with Mangasarian original paper   
