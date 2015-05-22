function [RankedVariables,nbsvvec,values]=FeatSelregspanbound(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)

% Usage
%  
% [RankedVariables,nbsvvec,values]=FeatSelregspanbound(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)
%
%   rank variables according to their values on Span estimate when removed
%
%   x,y     : input data
%   c       : penalization of misclassified examples
%   kernel  : kernel type
%   kerneloption : kernel hyperparameters
%   verbose
%   FeatSeloption : structure containing FeatSeloption parameters
%           Fields           
%
%           AlphaApprox : O 
%           RemoveChunks : number of variable to remove (a number or 'half')
%           StopChunks   : remove 1 variable at a time when number of
%           variables reaches this value
%
%
% alain.rakoto@insa-rouen.fr
%   
%   \bibitem[Rakotomamonjy(2006)]{rakoto_featselreg}
%    A.~Rakotomamonjy.
%   \newblock Analysis of SVM regression bound for feature selection,
%   \newblock Neurocomputing 2006


if nargin <8
    FeatSeloption.AlphaApprox=1;
end;

%----------------------------------------------------------%
%              Testing Fields Existence                    % 
%----------------------------------------------------------%
if  isfield(FeatSeloption,'SEeta')
    eta=FeatSeloption.SEeta;
else
    eta=0.001;
end;
if ~isfield(FeatSeloption,'AlphaApprox')
    FeatSeloption.AlphaApprox=1;
end;
if ~isfield(FeatSeloption,'RemoveChunks')
    FeatSeloption.RemoveChunks=1;
end;
if ~isfield(FeatSeloption,'StopChunks')
    FeatSeloption.StopChunks=10;
end;
if strcmp(FeatSeloption.RemoveChunks,'half')
    half=1;
else
    half=0;
end;

lambd=1e-8;
lambdaregul=1e-13;
caux=diag((1/c)*ones(2*length(y),1));
SelectedVariables = [1:size(x,2)]; %list of remaining variable
EliminatedVariables = []; %list of eliminated variables
alphaall=[];
values=[];
nbsvvec=[];

while length(SelectedVariables)~=1
    
    if half==1
        FeatSeloption.RemoveChunks=round(length(SelectedVariables)/2);
    end;
    
    
    if FeatSeloption.RemoveChunks<=FeatSeloption.StopChunks/2 & half == 1
        FeatSeloption.RemoveChunks=1;
    end;
    
    if length(SelectedVariables)<=FeatSeloption.StopChunks
        FeatSeloption.RemoveChunks=1;
    end;
    
    xaux=x(:,SelectedVariables);
    

    caux1=diag((1/c)*ones(length(y),1)); 
    
    %------------------------------------------------------------------	
    ps=svmkernel(xaux,kernel,kerneloption);
    n=size(xaux,1);
    I = eye(n);
    Idif = [I -I];
    H = Idif'*ps*Idif + caux;
    ee = [-epsilon+y ; -epsilon-y]; % [ alpha*   alpha]
    A = [-ones(1,n)  +ones(1,n) ]';
    b=0;   
    Cinf=inf;
    [alphatemp,bias,posalpha]=monqp(H,ee,A,b,Cinf,lambd,verbose,x,ps);
    nbsv=length(alphatemp);
    alphaall=zeros(length(H),1);
    alphaall(posalpha)=alphatemp;
    AlphaStar=alphaall(1:n);
    Alpha=alphaall(n+1:end);
    %-------------------------------------------------------------------
    
    
    
    T=[];
    alphatemp=alphaall;
    for i=1:length(SelectedVariables)
        SelectVariablesAux=SelectedVariables;
        if FeatSeloption.AlphaApprox
                  
            
            
            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption); 
            alphaaux=alphatemp;
            nbsvaux=length(find(alphatemp));
            AlphaStarTemp=alphaaux(1:n);
            AlphaTemp=alphaaux(n+1:end);
            
            %----------------------------------------------------------
            %            Calcul des nouveaux St^2
            newpos=find(alphaaux(1:n)>0|alphaaux(n+1:2*n)> 0);
            knon=psnon(newpos,newpos) +caux1(newpos,newpos);
            D=(eta./(AlphaTemp(newpos)+AlphaStarTemp(newpos)));
            ksvaux=[knon ones(nbsvaux,1)];
            ksvaux=[ksvaux; [ones(1,nbsvaux) 0]];
            ksvaux=ksvaux+diag([D;0]);
            sp2aux=1./diag(inv(ksvaux+lambdaregul*eye(size(ksvaux))));
            sp2aux=sp2aux(1:nbsvaux)-D;
            T(i)= (AlphaStarTemp(newpos)+AlphaTemp(newpos))'*sp2aux;   % Ordre 0  Borne de l'erreur LOO
            
        else
            

            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption);
            Hnon=Idif'*psnon*Idif + caux;
            % calcul des nouveaux AlphaStar et Alpha
            A = [ones(1,n)  -ones(1,n) ]';
            b=0;   
            Cinf=inf;
            [alphatemp , lambda , pos] =  monqpCinfty(Hnon,ee,A,b,lambd,verbose,x,psnon,alphaall); 
            nbsvaux=length(alphatemp);
            alphaaux=zeros(size(ee));
            alphaaux(pos)=alphatemp;
            AlphaStarTemp=alphaaux(1:n);
            AlphaTemp=alphaaux(n+1:end);
            
            %----------------------------------------------------------
            %            Calcul des nouveaux St^2
            newpos=find(alphaaux(1:n)>0|alphaaux(n+1:2*n)> 0);
            knon=psnon(newpos,newpos) +caux1(newpos,newpos);
            D=(eta./(AlphaTemp(newpos)+AlphaStarTemp(newpos)));
            ksvaux=[knon ones(nbsvaux,1)];
            ksvaux=[ksvaux; [ones(1,nbsvaux) 0]];
            ksvaux=ksvaux+diag([D;0]);
            sp2aux=1./diag(inv(ksvaux+lambdaregul*eye(size(ksvaux))));
            sp2aux=sp2aux(1:nbsvaux)-D;
            T(i)= (AlphaStarTemp(newpos)+AlphaTemp(newpos))'*sp2aux;   % Ordre 0  Borne de l'erreur LOO
            
            
            
        end;
    end
    
    
    
    [nointerest indiceDJ] = sort(T);
    EliminatedVariables = [SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) EliminatedVariables];
    values= [T(indiceDJ(1:FeatSeloption.RemoveChunks)) values];
    SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) = [];
    
end;

RankedVariables=[SelectedVariables EliminatedVariables ];
values=[T(end) values];


