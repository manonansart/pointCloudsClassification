function [xsup,w,b,nbsv,classifier,pos,obj]=svmmulticlassoneagainstone(x,y,nbclass,c,epsilon,kernel,kerneloption,verbose,warmstart);
%[xsup,w,b,nbsv,classifier,posSigma]=svmmulticlassoneagainstone(x,y,nbclass,c,epsilon,kernel,kerneloption,verbose);
%
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
% classifier gives which class against which one
%


%
% See svmclass, svmmultival
%
%


% 29/07/2000 Alain Rakotomamonjy


if nargin < 8
    verbose=0;
end;


xsup=[];  % 3D matrices can not be used as numebr of SV changes
w=[];
b=[];
pos=[];
SigmaOut=[];
span=1;
classifier=[];
nbsv=zeros(1,nbclass);
nbsuppvector=zeros(1,nbclass);
k=1;
if isempty(x) & strcmp(kernel,'numerical') & isfield(kerneloption,'matrix')
    Kaux=kerneloption.matrix;
    kernelparam=1;
    xapp=[];
else
    kernelparam=0;
end;
obj=0;
for i=1:nbclass-1
    for j=i+1:nbclass
        indi=find(y==i);
        indj=find(y==j);
        yone=[ones(length(indi),1);-ones(length(indj),1)];
        if ~isempty(x)
            xapp=[x(indi,:); x(indj,:)];
        end;
        if exist('warmstart') & isfield(warmstart,'nbsv');
            nbsvinit=cumsum([0 warmstart.nbsv]);
            alphainit=zeros(size(yone));
                    aux=[indi;indj];
            posaux=find(ismember(aux,warmstart.pos(nbsvinit(k)+1:nbsvinit(k+1))));
            alphainit(posaux)= abs(warmstart.alpsup(nbsvinit(k)+1:nbsvinit(k+1)));
        else
            alphainit=[];
        end;
        if kernelparam==1;
            if size(indi,1)==1 && size(indi,2)>1
                kerneloption.matrix=Kaux([indi indj],[indi indj]);
            else
                kerneloption.matrix=Kaux([indi  ;indj],[indi; indj]);
            end
        end;

        [xsupaux,waux,baux,posaux,timeaux,alphaaux,objaux]=svmclass(xapp,yone,c,epsilon,kernel,kerneloption,verbose,span,alphainit);

        [n1,n2]=size(waux);
        nbsv(k)=n1;
        classifier(k,:)=[i j];
        xsup=[xsup;xsupaux];
        w=[w;waux];
        b=[b;baux];
        aux=[indi;indj];
        pos=[pos;aux(posaux)];
        obj=obj+objaux;
        k=k+1;
    end;
end;

