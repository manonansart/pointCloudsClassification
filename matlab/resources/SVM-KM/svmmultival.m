function [ypred,maxi,ypredMat]=svmmultival(x,xsup,w,b,nbsv,kernel,kerneloption)

% USAGE ypred=svmmultival(x,xsup,w,b,nbsv,kernel,kerneloption)
% 
% Process the class of a new point x of a one-against-all 
% or a all data at once MultiClass SVM
% 
% This function should be used in conjuction with the output of
% svmmulticlass.
%
%
% See also svmmulticlass, svmval
%

% 29/07/2000 Alain Rakotomamonjy


[n1,n2]=size(x);
nbclass=length(nbsv);
y=zeros(n1,nbclass);
nbsv=[0 nbsv];
aux=cumsum(nbsv);
for i=1:nbclass
    if ~isempty(xsup)
         xsupaux=xsup(aux(i)+1:aux(i)+nbsv(i+1),:);
        waux=w(aux(i)+1:aux(i)+nbsv(i+1));
        baux=b(i);
        ypred(:,i)= svmval(x,xsupaux,waux,baux,kernel,kerneloption);
    else
      if isempty(x)  %  Kernel matrix is given as a parameter 
        waux=w(aux(i)+1:aux(i)+nbsv(i+1));
        baux=b(i);
        kernel='numerical';
        xsupaux=[];
        pos=aux(i)+1:aux(i)+nbsv(i+1);
        kerneloption2.matrix=kerneloption.matrix(:,pos);
        ypred(:,i)= svmval(x,xsupaux,waux,baux,kernel,kerneloption2);
      end;
    end;
   
end;
ypredMat=ypred;
[maxi,ypred]=max(ypred');
maxi=maxi';
ypred=ypred';
