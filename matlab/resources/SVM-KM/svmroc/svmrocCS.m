function [xsup,w,w0,pos,timeps,alpha,matriceind]=svmrocCS(x,y,C,kppv,margin,lambda,kernel,kerneloption,verbose,span)
 
% USAGE
%
% [xsup,w,w0,pos,timeps,alpha,matriceind]=svmrocCS(x,y,C,kppv,margin,lambda,kernel,kerneloption,verbose,span)
%
% x and y are the learning set
% 
% 
% SVM ROC curve optimizer. This code can only deal with a few kppv because otherwise,
% the number of constraints becomes too large and the QP problem becomes intractable
% due to memory limitation. 
% For this reason, it should be used only for case study.
%
%

% 30/07/2004 A. Rakotomamonjy

indpos=find(y==1);
indneg=find(y==-1);
nbpos=length(indpos);
nbneg=length(indneg);
timeps=0;
k=1;

% This is for having all the couples in the matrice ind matrix
% ell=nbneg*nbpos;
% k=1;
% matriceind=[];
% for n=0:ell-1
%         j=mod(n,nbneg)+1 ;%+ nbneg*(mod(n,nbneg)==0);
%         i=floor(n/nbneg)+1;
%         
%         matriceind(k,:)=[i j];    
%         k=k+1;
%         
% end;

%-------------------------------------------------------------------------
%
% dist in feature space of each positive example to negative examples
%
% the idea here is to select a subset of couple for
% optimizing the ranking
%
% 


for i=1:nbpos+nbneg
    norme2(i)=svmkernel(x(i,:),kernel,kerneloption);
end;

matriceindneg=[];
% select only the k-nearest positive neighbor of negative xamples
for i=1:nbneg
    aux1=svmkernel(x(indneg(i),:),kernel,kerneloption,x(indpos,:));
    dist=norme2(indneg(i))*ones(1,nbpos) + norme2(indpos) - 2*aux1 ;
    [aux,indicesorted]=sort(dist');
    minim=min(length(indicesorted),kppv);
    matriceindneg=[matriceindneg; i*ones(minim,1) indicesorted(1:minim)];
end;
vect=unique(matriceindneg(:,2)); 
% process only these couples of nn of these positive samples.
dist=[];
matriceind=[];
for i=1:length(vect)
    aux1=svmkernel(x(indpos(vect(i)),:),kernel,kerneloption,x(indneg,:));
    dist=norme2(indpos(vect(i)))*ones(1,nbneg) + norme2(indneg) - 2*aux1 ;
    [aux,indicesorted]=sort(dist');
    minim=min(length(indicesorted),kppv);
    matriceind=[matriceind; vect(i)*ones(minim,1) indicesorted(1:minim)];
end;
taille=length(matriceind);

%-----------------------------------------------------------------------


if verbose
    fprintf('Positive : %d  \t Negative : %d size of QP : %d \n',nbpos,nbneg, taille)
end;
%-----------------------------------------------------------------------

% Matrix preparation for solving the problem

% for n=1:taille
%     i=matriceind(n,1);
%     j=matriceind(n,2);
%     for m=1:taille
% 
%     k=matriceind(m,1);
%     l=matriceind(m,2);
%       K(n,m)= svmkernel(x(indpos(i),:),kernel,kerneloption,x(indpos(k),:))-svmkernel(x(indpos(i),:),kernel,kerneloption,x(indneg(l),:))...
%            -svmkernel(x(indneg(j),:),kernel,kerneloption,x(indpos(k),:))+svmkernel(x(indneg(j),:),kernel,kerneloption,x(indneg(l),:));
%             
%     
%            
%     end;
% end;
%keyboard

K=svmkernel(x(indpos(matriceind(:,1)),:),kernel,kerneloption)-svmkernel(x(indpos(matriceind(:,1)),:),kernel,kerneloption,x(indneg(matriceind(:,2)),:))...
   -svmkernel(x(indneg(matriceind(:,2)),:),kernel,kerneloption,x(indpos(matriceind(:,1)),:))+svmkernel(x(indneg(matriceind(:,2)),:),kernel,kerneloption);
    



lambda=1e-7;
f=ones(length(K),1)*margin;
[alpha, lambda, pos]=monqp(K,f,zeros(taille,1),0,C,lambda,verbose);
w=alpha;
w0=0;
span=1;
xsuppos=x(indpos(matriceind(pos,1)),:);
xsupneg=x(indneg(matriceind(pos,2)),:);
xsup={xsuppos xsupneg};
if verbose
    fprintf('\n\nPositive : %d  \t Negative : %d size of QP : %d \n',nbpos,nbneg, taille)
end;