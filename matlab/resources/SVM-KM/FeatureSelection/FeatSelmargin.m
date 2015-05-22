function [RankedVariables,nbsv,values]=FeatSelmargin(x,y,c,kernel,kerneloption,verbose,span,FeatSeloption)

% Usage
%  
% [RankedVariables,nbsv,values]=FeatSelmargin(x,y,c,kernel,kerneloption,verbose,span,FeatSeloption)
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
%           AlphaApprox : O for retraining , 1 for approximation 
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

caux=diag((1/c)*ones(length(y),1));
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
    ps=svmkernel(xaux,kernel,kerneloption);
    lambd=1e-7;
    psc=ps+caux;
    H =psc.*(y*y');
 
    e = ones(size(y));
    A = y;    b = 0;
    [alpha , lambda , pos] =  monqpCinfty(H,e,A,b,lambd,verbose,x,psc,alphaall); 

    alphaall=zeros(size(e));
    alphaall(pos)=alpha;
    nbsv=[length(pos) nbsv];
    
  
    SelectVariablesAux=SelectedVariables;
    w2=[];
    alphatemp=alpha;

    for i=1:length(SelectedVariables)
        
          
        SelectVariablesAux=SelectedVariables;  
        
        if FeatSeloption.AlphaApprox
         
            caux1=caux(pos,pos);
             SelectVariablesAux(i)=[];
            xnon2 = x(pos,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption)+caux1;
            
            %------------------------------------------------------------
            Hnon=psnon.*(y(pos)*y(pos)');
            w2aux=(-0.5*alphaall(pos)'*Hnon*alphaall(pos) +e(pos)'*alphaall(pos));
            w2(i)=w2aux;
            
        else

            SelectVariablesAux(i)=[];
            xnon2 = x(:,SelectVariablesAux);
            psnon=svmkernel(xnon2,kernel,kerneloption)+caux;
            
            
            Hnon=psnon.*(y*y');
            [alphatemp , lambda , pos] =  monqpCinfty(Hnon,e,A,b,lambd,verbose,x,psnon,alphatemp); 
            alphaaux=zeros(size(e));
            alphaaux(pos)=alphatemp;
            w2aux=(-0.5*alphaaux(pos)'*Hnon(pos,pos)*alphaaux(pos) +e(pos)'*alphaaux(pos));
            w2(i)=w2aux;
        end;
    end
    
    [nointerest indiceDJ] = sort(w2);
    EliminatedVariables = [SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) EliminatedVariables];
    values= [w2(indiceDJ(1:FeatSeloption.RemoveChunks)) values];
    SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) = [];
    
end;

RankedVariables=[ EliminatedVariables ];

