function [RankedVariables,nbsv,values]=FeatSelmargin(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)

% Usage
%  
% [RankedVariables,nbsv,values]=FeatSelmargin(x,y,c,kernel,kerneloption,verbose,FeatSeloption)
%
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


lambd=1e-8;
if nargin <8
    FeatSeloption.AlphaApprox=1;
end;

%----------------------------------------------------------%
%              Testing Fields Existence                    % 
%----------------------------------------------------------%

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

% caux=diag((1/c)*ones(length(y),1));

caux=diag((1/c)*ones(length(y)*2,1));  % REGRESSION

SelectedVariables = [1:size(x,2)]; %list of remaining variable
EliminatedVariables = []; %list of elimanted variables
alphaall=[];
betaall=[];
nbsv=[];
values=[];
while length(SelectedVariables)~=0
    length(SelectedVariables);
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
    
    %------------------------------------------------------------------	
    ps=svmkernel(xaux,kernel,kerneloption);
    n=size(xaux,1);
    I = eye(n);
    Idif = [I -I];
    H = Idif'*ps*Idif + caux;
    c = [-epsilon+y ; -epsilon-y]; % [ alpha*   alpha]
    A = [-ones(1,n)  +ones(1,n) ]';
    b=0;   
    Cinf=inf;
    [alpha,bias,pos]=monqp(H,c,A,b,Cinf,lambd,verbose,x,ps);
    alphaall=zeros(length(H),1);
    alphaall(pos)=alpha;
    %-------------------------------------------------------------------
    
    SelectVariablesAux=SelectedVariables;
    w2=[];
    alphatemp=alphaall;
    
    for i=1:length(SelectedVariables)
        
       
        
        SelectVariablesAux=SelectedVariables;  
        
        if FeatSeloption.AlphaApprox 
            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption);
            Hnon=Idif'*psnon*Idif + caux;
            
            % this is the objective value of the dual hence it is equal to
            % the objective value of the primal which is the modified w^2
           w2aux=alphaall(pos)'*Hnon(pos,pos)*alphaall(pos);
           w2(i)=w2aux;

            
        else
            
            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption);
            
            Hnon=Idif'*psnon*Idif + caux;
            [alphatemp , lambda , pos] = monqp(Hnon,c,A,b,Cinf,lambd,verbose,x,psnon,alphaall); 
            alphaaux=zeros(size(c));
            alphaaux(pos)=alphatemp;
            w2aux=alphaaux(pos)'*Hnon(pos,pos)*alphaaux(pos);
            w2(i)=w2aux;
            
        end;
    end
    
    [nointerest indiceDJ] = sort(w2);
    EliminatedVariables = [SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) EliminatedVariables];
    values= [w2(indiceDJ(1:FeatSeloption.RemoveChunks)) values];
    SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) = [];
    
end;

RankedVariables=[ EliminatedVariables ];

