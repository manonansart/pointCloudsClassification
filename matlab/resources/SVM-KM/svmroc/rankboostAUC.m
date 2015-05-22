
function [alpha,threshold,rankfeat]=rankboostAUC(xapp,yapp,T);

%  USAGE
%
% [alpha,threshold,rankfeat]=rankboostAUC(xapp,yapp,T);
%
%  This a Rankboost algorithm as described in the freund et al
%  Journal of Machine Learning Research paper.
%  
%  xapp and yapp are the learning set data and labels
%  T is the number of weak learners which is a step function.
%
%  the outputs are 
%  
%   alpha  : vector  of weigth of each weak learner
%   threshold : vector of each weak learner translation
%   rankfeat : vector of each weak learner feature.
%
%  see also rankboostAUCval

% 30/07/2004  A. Rakotomamonjy
%
%



[n,dim]=size(xapp);
indpos=find(yapp==1);
npos=length(indpos);
indneg=find(yapp==-1);
nneg=length(indneg);
yapp=[yapp(indpos);yapp(indneg)];    % on remet les données en ordre!
xapp=[xapp(indpos,:);xapp(indneg,:)];
indpos=find(yapp==1);
indneg=find(yapp==-1);

% dans les vecteurs on stocke d'abord les v(xo) puis les v(x1) xo c'est les
% pos et x1 les neg par rapport à yapp (voir papier de freund ou on cherche
% h(xo)>h(x1)
%v=ones(n,1)/sqrt(npos*nneg); %v=[v(xo);v(x1)]
v=[1/npos*ones(npos,1);1/nneg*ones(nneg,1)];% Voir Papier JMLR Algo Rankboost.B
s=-yapp; % s=[s(xo);s(x1)];
% D


% pos : xo    neg x1
%keyboard
for k=1:T
    
    
    fprintf('.');
    vpos=sum(v(indpos)); % Equ 6. du proc ICML 
    vneg=sum(v(indneg)); %  Equ 6. du proc ICML 
    d=v .*[vneg*ones(npos,1);vpos*ones(nneg,1)]; 
    mpi= s.*d; % Definition de pi(x) Paragraphe en dessous equ. 8
    
    
    %   WEAK_LEARNER ------------------------------------------
    %   as described in the JMLR paper
    % les ranking features c'est les dimensions
    rmax=0;
    for i=1:dim % on regarde les ranking feature
        fi=xapp(:,i);
        [sortedfi,indicesorted]=sort(fi) ;% les ranking sont triés ... Sorting features
        sortedfi=flipud(sortedfi); % par ordre decroissant 
        candthreshold=[inf;sortedfi- min(abs(diff(sortedfi)))/2]; % candidate threshold = inf U ranked feature + chouia 
        J=length(sortedfi);
        L=0;
        % On suppose dans la suite que qdef= 0 donc on n'a pas besoin de R
        for j=2:J+1; 
         ind=find(fi <=candthreshold(j-1) & fi >candthreshold(j) );
            L=L + sum(mpi(ind));
            if abs(L)> abs(rmax)
                rmax=L;
                rankfeat(k)=i;
                threshold(k)=candthreshold(j);
                
            end;
            
        end;
    end;
    %------------------------------------------------------------
    % This is for avoiding numerical error if rmax=-1 or 1 which occurs in
    % case of perfect ranking
    if abs(abs(rmax)-1) < 0.00001
        rmax=sign(rmax)*0.99999;
    end;    
    alpha(k)=0.5*log((1+rmax)/(1-rmax));
    % ----------------- UPDATE De V
    v1=v(indneg).*exp(-alpha(k)*(xapp(indneg,rankfeat(k))>threshold(k)));
    v0=v(indpos).*exp(alpha(k)*(xapp(indpos,rankfeat(k))>threshold(k)));
    if sum(v0)==0 || sum(v1)==0
        v=[v0;v1];
    else
        v=[v0/sum(v0);v1/sum(v1)];
    end;    
end;

alpha=-alpha; % this is due to the fact that s=-yapp so the sign of alpha makes
% the positive examples negative!!  
