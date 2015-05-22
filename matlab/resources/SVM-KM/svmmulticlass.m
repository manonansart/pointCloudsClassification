function [xsup,w,b,nbsv,pos,alpha]=svmmulticlass(x,y,nbclass,C,epsilon,kernel,kerneloption,verbose, alphainit)


% USAGE 
%[xsup,w,b,nbsv,pos,alpha]=svmmulticlass(x,y,nbclas,C,epsilon,kernel,kerneloption,verbose, alphainit)
%
% Support vector machine for multiclass CLASSIFICATION
% This routine classify the training set with a support vector machine
% using quadratic programming algorithm (active constraints method)
%
% INPUT
%
% Training set
%      x  		: input data 
%      y  		: output data
% parameters
%		c		: Bound on the lagrangian multipliers     
%		lambda		: Conditioning parameter for QP method
%		kernel		: kernel  type. classical kernel are
%
%		Name			parameters
%		'poly'		polynomial degree
%		'gaussian'	gaussian standard deviation
%
%		for more details see svmkernel
% 
%		kerneloption : parameters of kernel
%
%		for more details see svmkernel
%
% 		verbose : display outputs (default value is 0: no display)
%
%       alphainit : initialization vector of QP problem
%
% OUTPUT
%
% xsup	coordinates of the Support Vector
% w      weight
% b		bias
% pos    position of Support Vector
% alpha  Lagragian multiplier
%
%
% see also svmreg, svmkernel, svmval

%	06/01/2003 Alain Rakotomamonjy
%
%       scanu@insa-rouen.fr, alain.rakoto@insa-rouen.fr


if nargin< 7
    alphainit=[];
end;


if nargin < 6
    verbose = 0;
end

if nargin < 5
    kerneloption = 1;
end

if nargin < 4
    kernel = 'gaussian';
end

if nargin < 3
    lambda = 0.000000001;
end

if nargin < 2
    C = 100000;
end

%------------------------------------------------------
% initialisation
%-------------------------------------------------------
[y,ind]=sort(y);
x=x(ind,:);

yextended=repmat(y,nbclass,1);
xextended=repmat(x,nbclass,1);


ell=size(x,1);
n=sum(y==1);

if size(C,1)==1
    C=C*ones(size(y));
end;

%------------------------------------------------------
% construction des matrices associé au pb QP
%-------------------------------------------------------


MatAux=zeros(ell,ell);
MatAuxQ3=zeros(ell,ell);
Q21=zeros(ell*nbclass,ell*nbclass);
Q22=zeros(ell*nbclass,ell*nbclass);
debut=1;
debutQ3=1;
for s=1:nbclass
    indclass=find(y==s);
    ellinclass=length(indclass);
    MatAux(debut:debut+ellinclass-1,debut:debut+ellinclass-1)=svmkernel(x(indclass,:),kernel,kerneloption);
    debut=debut+ellinclass;
    
    MatAuxQ3(debutQ3:debutQ3+ell-1,debutQ3:debutQ3+ell-1)=svmkernel(x,kernel,kerneloption);
    debutQ3=debutQ3+ell;   
    for j=1:ell
        Kij=svmkernel(x,kernel,kerneloption,x(j,:));
        yj=y(j);
        Q21( (yj-1)*ell +1: yj *ell, (s-1)*ell + j )= -Kij; 
        
        ind=find(yextended==s);
        Q22(ind,(s-1)*ell + j )= - svmkernel(xextended(ind,:),kernel,kerneloption,x(j,:));
        
    end;
end;

Q1=repmat(MatAux,nbclass,nbclass);
Q3=MatAuxQ3;
Q2=Q21+Q22;

Q=Q1+Q2+Q3;


%------------------------------------------------------
% Les contraintes
%-------------------------------------------------------



yii=[];
Am=zeros(nbclass, size(yextended,1));
Am1=zeros(nbclass, size(yextended,1));
for s=1:nbclass
    ind=(s-1)*ell+1: s*ell;
    Am(s,ind)=ones(1,length(ind));
    ind1=find(yextended==s);
    Am1(s,ind1)=ones(1,length(ind1));
    yii= [yii;y==s];
end;

A=Am-Am1;

c=2*ones(size(yextended));
b=zeros(size(A,1),1);
Cvect=repmat(C,3,1);
unused=find(yii==1);
Cvect(unused)=0.00000*ones(length(unused),1);




xinit=zeros(size(Cvect));
xinit(n+1)=Cvect(n+1)/2;
[xnew, lambda, pos] = monqp(Q,c,A',b,Cvect,epsilon,verbose,x,[],xinit);


b=-lambda;
alpha=zeros(size(Cvect));
alpha(pos)=xnew;
w=zeros(size(xextended,1),1);
xsup=[];


for s=1:nbclass
    for i=1:ell
        if y(i)==s
            for m=1:nbclass
                w((s-1)*ell+i)=w((s-1)*ell+i)+alpha((m-1)*ell+i);
            end;
        end;
        w((s-1)*ell+i)=w((s-1)*ell+i)-alpha((s-1)*ell+i);
    end;
end;

waux=[];
for s=1:nbclass
    ind=find(w((s-1)*ell+1:(s-1)*ell+ell)~=0);
    waux=[waux;w( (s-1)*ell+ ind)];
    nbsv(s)=length(ind);
    xsup=[xsup; x(ind,:)]; 
end;
w=waux;