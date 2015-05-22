function [RankedVariables,values]=FeatSelregr2w2(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)

% Usage
%  
%  [RankedVariables,values]=FeatSelregr2w2(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)
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
caux=diag((1/c)*ones(length(y),1));
SelectedVariables = [1:size(x,2)]; %list of remaining variable
EliminatedVariables = []; %list of elimanted variables

alphaall=[];
betaall=[];
values=[];
caux=diag((1/c)*ones(length(y)*2,1));  % REGRESSION
while length(SelectedVariables)~=0
    
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
    ps=svmkernel(xaux,kernel,kerneloption);
    lambd=1e-7;
    
    
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
%     [alpha,bias,posalpha]=monqp(H,ee,A,b,Cinf,lambd,verbose,x,ps);
%     alphaall=zeros(length(H),1);
%     alphaall(posalpha)=alpha;
    %-------------------------------------------------------------------
    
    
    
    
    
    %---------------------------------------------%
    %  calcul de r2                               %  
    %---------------------------------------------%
    caux1=diag((1/c)*ones(length(y),1)); 
    psc=ps+caux1;

    
    
   kerneloptionr2.matrix=psc;
   [betaall,r2,posbeta]= r2smallestsphere([],[],kerneloptionr2);
    
    
    
    r2w2=[];
%    alphatemp=alpha;
%    betatemp=beta;
    for i=1:length(SelectedVariables)
        
        SelectVariablesAux=SelectedVariables;
        
        if FeatSeloption.AlphaApprox
            %caux1=caux(pos,pos);
            
            
            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption);
            
            
            Daux=diag(psnon+caux1);
            Hnon=Idif'*psnon*Idif + caux;;
            
            
            sumalpha=sum(alphaall);
            r2aux=-betaall(posbeta)'*psnon(posbeta,posbeta)*betaall(posbeta)+Daux(posbeta)'*betaall(posbeta);
            r2w2(i)=r2aux*sumalpha;
            
        else
            
            
            
            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption);
            
            Hnon=Idif'*psnon*Idif + caux;
            
            
            % calcul de w^2
            A = [ones(1,n)  -ones(1,n) ]';
            b=0;   
            Cinf=inf;
            [alphatemp , lambda , pos] =  monqpCinfty(Hnon,ee,A,b,lambd,verbose,x,psnon,alphaall); 
            alphaaux=zeros(size(ee));
            alphaaux(pos)=alphatemp;
            sumalpha=sum(alphatemp);
            % calcul de R^2
            psnon=psnon+caux1;
            D=diag(psnon);
            A = ones(size(D));
            b=1;
            verbose=0;
            C=inf;
            [betatemp,lagrangian,posbeta]=monqp(2*psnon,D,A,b,C,lambd,verbose,x,psnon,betaall);
            betaaux=zeros(size(D));
            betaaux(posbeta)=betatemp;
            r2aux=-betaaux'*psnon*betaaux+D'*betaaux;
         
            r2w2(i)=r2aux*sumalpha;
        end;
        
    end
    
    [nointerest indiceDJ] = sort(r2w2);
    EliminatedVariables = [SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) EliminatedVariables];
    values= [r2w2(indiceDJ(1:FeatSeloption.RemoveChunks)) values];
    SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) = [];
end;

RankedVariables=[SelectedVariables EliminatedVariables ];


