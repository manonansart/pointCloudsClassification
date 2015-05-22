function [RankedVariables,values]=FeatSelr2w2(x,y,c,kernel,kerneloption,verbose,span,FeatSeloption)

% Usage
%  
%   [RankedVariables,values]=FeatSelr2w2(x,y,c,kernel,kerneloption,verbose,span,FeatSeloption)
%
%
%
%   x,y     : input data
%   c       : penalization of misclassified examples
%   kernel  : kernel type
%   kerneloption : kernel hyperparameters
%   verbose
%   span    : matrix for semiparametric learning
%   FeatSeloption : structure containing FeatSeloption parameters
%           Fields           
%
%           AlphaApprox : O for retraining 1, for approximation 
%           RemoveChunks : number of variable to remove (a number or 'half')
%           StopChunks   : remove 1 variable at a time when number of variables reaches this value
%           FirstOrderMethod : how to calculate the derivatives
%               'grad','scal', 'absgrad', 'absscal'
%
%
%
% alain.rakoto@insa-rouen.fr
%   
%   \bibitem[Rakotomamonjy(2002)]{rakoto_featsel}
%    A.~Rakotomamonjy.
%   \newblock Variable selection using svm based criteria.
%   \newblock Technical Report 02-004, Insa de Rouen Perception Syst\`eme
%   Informations, http://asi.insa-rouen.fr/\char126arakotom, 2002.
%


%

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
    psc=ps+caux;
    
    H =psc.*(y*y');
    e = ones(size(y));
    A = y;    b = 0;
    [alpha , lambda , posalpha] =  monqpCinfty(H,e,A,b,lambd,verbose,x,psc,alphaall); 

    alphaall=zeros(size(e));
    alphaall(posalpha)=alpha;
    
    %---------------------------------------------%
    %  calcul de r2                               %  
    %---------------------------------------------%
    
    D=diag(psc);
    A = ones(size(D));
    b=1;
    verbose=0;
    C=inf;
    [beta,r2,posbeta]=monqp(2*psc,D,A,b,C,lambd,verbose,x,psc,betaall);
    betaall=zeros(size(D));
    betaall(posbeta)=beta;
    pos=union(posalpha,posbeta);
    
 

    
    r2w2=[];
    alphatemp=alpha;
    betatemp=beta;
    for i=1:length(SelectedVariables)
        
        SelectVariablesAux=SelectedVariables;
        
        if FeatSeloption.AlphaApprox
             caux1=caux(pos,pos);
             

            SelectVariablesAux(i)=[];
            xnon2 = x(pos,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption)+caux1;


            Daux=diag(psnon);
            Hnon=psnon.*(y(pos)*y(pos)');
            
            w2aux=2*(-0.5*alphaall(pos)'*Hnon*alphaall(pos) +e(pos)'*alphaall(pos)); % chapelle p 16
            r2aux=-betaall(pos)'*psnon*betaall(pos)+Daux'*betaall(pos);
            r2w2(i)=r2aux*w2aux;
       
        else
            

            
            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption)+caux;
            
            Hnon=psnon.*(y*y');
            e = ones(size(y));
            A = y;
            b = 0;
            [alphatemp , lambda , pos] =  monqpCinfty(Hnon,e,A,b,lambd,verbose,x,psnon,alphatemp); 
            alphaaux=zeros(size(e));
            alphaaux(pos)=alphatemp;
            w2aux=(-0.5*alphaaux(pos)'*Hnon(pos,pos)*alphaaux(pos) +e(pos)'*alphaaux(pos));
            D=diag(psnon);
            A = ones(size(D));
            b=1;
            verbose=0;
            C=inf;
            [betatemp,r2,posbeta]=monqp(2*psnon,D,A,b,C,lambd,verbose,x,psnon,betatemp);
            betaaux=zeros(size(D));
            betaaux(posbeta)=betatemp;
            r2aux=-betaaux'*psnon*betaaux+D'*betaaux;
            r2w2(i)=r2aux*w2aux;
        end;
        
    end

    [nointerest indiceDJ] = sort(r2w2);
    EliminatedVariables = [SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) EliminatedVariables];
    values= [r2w2(indiceDJ(1:FeatSeloption.RemoveChunks)) values];
    SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) = [];
end;

RankedVariables=[SelectedVariables EliminatedVariables ];






%         SelectVariablesAux=SelectedVariables;
%         SelectVariablesAux(i)=[];
%         psnon= svmkernel(x(pos,SelectVariablesAux),kernel,kerneloption);
