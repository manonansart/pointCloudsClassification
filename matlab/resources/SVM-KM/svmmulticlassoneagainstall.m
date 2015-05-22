function [xsup,w,b,nbsv,pos,obj]=svmmulticlass(x,y,nbclass,c,epsilon,kernel,kerneloption,verbose,warmstart);

% USAGE [xsup,w,b,nbsv,pos,obj]=svmmulticlass(x,y,nbclass,c,epsilon,kernel,kerneloption,verbose);
%
%
% SVM Multi Classes Classification One against Others algorithm
%
% y is the target vector which contains integer from 1 to nbclass.
% 
% This subroutine use the svmclass function
% 
% the differences lies in the output nbsv which is a vector
% containing the number of support vector for each machine
% learning.
% For xsup, w, b, the output of each ML are concatenated
% in line.
% 
% 
% See svmclass, svmmultival
%
%
% 29/07/2000 Alain Rakotomamonjy


xsup=[];  % 3D matrices can not be used as numebr of SV changes
w=[];
b=[];
pos=[];
span=1;
qpsize=1000;
nbsv=zeros(1,nbclass);
nbsuppvector=zeros(1,nbclass);
obj=0;

for i=1:nbclass
    
    yone=(y==i)+(y~=i)*-1;
    if exist('warmstart') & isfield(warmstart,'nbsv');
        nbsvinit=cumsum([0 warmstart.nbsv]);
        alphainit=zeros(size(yone));
        alphainit(warmstart.pos(nbsvinit(i)+1:nbsvinit(i+1)))= abs(warmstart.alpsup(nbsvinit(i)+1:nbsvinit(i+1)));
    else
        alphainit=[];
    end;
    if size(yone,1)>4000
        [xsupaux,waux,baux,posaux]=svmclassls(x,yone,c,epsilon,kernel,kerneloption,verbose,span,qpsize,alphainit);
    else
        [xsupaux,waux,baux,posaux,timeaux,alphaaux,objaux]=svmclass(x,yone,c,epsilon,kernel,kerneloption,verbose,span,alphainit);
    end;
    
    nbsv(i)=length(posaux);
    xsup=[xsup;xsupaux];
    w=[w;waux];
    b=[b;baux];
    pos=[pos;posaux];
    obj=obj+objaux;
end;


