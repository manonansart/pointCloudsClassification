function [ypred,vote]=svmmultivaloneagainstone(x,xsup,w,b,nbsv,kernel,kerneloption)

% USAGE [ypred,vote]=svmmultivaloneagainstone(x,xsup,w,b,nbsv,kernel,kerneloption,Sigma)
% 
% Process the class of a new point x.
% 
% This function should be used in conjuction with the output of
% svmmulticlassoneagainstone
%
%

%  modified : AR 07/03/2003  Added Sigma for dealing with particular scaling
%
% 29/07/2000 Alain Rakotomamonjy

if exist('Sigma')  ==1
    if ~strcmp(kernel,'gaussian') & ~(kerneloption==1)
    warning('Adaptive scaling is available only for gaussian kernel...');
    end
end;





if ~isempty(x)
    [n1,n2]=size(x);
    kernelparam=0;
else
    n1=size(kerneloption.matrix,1);
end;
nbclass=(1+ sqrt(1+4*2*length(nbsv)))/2;
vote=zeros(n1,nbclass);
nbsv=[0 nbsv];
aux=cumsum(nbsv);
k=1;
for i=1:nbclass-1;
    for j=i+1:nbclass;
        if nargin < 8;
            if ~isempty(xsup)
                xsupaux=xsup(aux(k)+1:aux(k)+nbsv(k+1),:);
                xaux=x;
                
            end;
            if isempty(xsup) & strcmp(kernel,'numerical') & isfield(kerneloption,'matrix')
                kerneloptionm.matrix=kerneloption.matrix(:,aux(k)+1:aux(k)+nbsv(k+1));
                kernelparam=1;
                xapp=[];
            end;            
        else    
            xsupaux=xsup(aux(k)+1:aux(k)+nbsv(k+1),:).*repmat(Sigma(i,:),nbsv(k+1),1);
            xaux= x.*repmat(Sigma(i,:),n1,1);
        end;
        waux=w(aux(k)+1:aux(k)+nbsv(k+1));
        baux=b(k);
        if ~kernelparam
        ypred= svmval(xaux,xsupaux,waux,baux,kernel,kerneloption);
        else
            ypred= svmval([],[],waux,baux,kernel,kerneloptionm);
        end;
        indi=find(ypred>=0);
        indj=find(ypred<0);
        vote(indi,i)=vote(indi,i)+1;
        vote(indj,j)=vote(indj,j)+1;
        k=k+1;
    end;
end;
[maxi,ypred]=max(vote');
ypred=ypred';