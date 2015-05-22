function [RankedVariables,nbsvvec,Values]=FeatSelr2w2(x,y,c,kernel,kerneloption,verbose,span,FeatSeloption)

% Usage
%  
%  [RankedVariables,nbsvvec,Values]=FeatSelr2w2(x,y,c,kernel,kerneloption,verbose,span,FeatSeloption)
%
%
%  Backward Variable ranking using sensivity of R^2w^2 to a variable as a criterion.
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
if ~isfield(FeatSeloption,'FirstOrderMethod')
    FeatSeloption.FirstOrderMethod='grad';
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
nbsvvec=[];
Values=[];
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
    ps=svmkernel(xaux,kernel,kerneloption);
    lambd=1e-7;
    psc=ps+caux;
    H =psc.*(y*y');
    e = ones(size(y));
    A = y;    b = 0;
    [alpha , lambda , posalpha] =  monqpCinfty(H,e,A,b,lambd,verbose,x,psc,alphaall); 
    nbsv=size(alpha,1);
    nbsvvec=[nbsvvec nbsv];
    alphaall=zeros(size(e));
    alphaall(posalpha)=alpha;
    w2=2*(-0.5*alpha'*H(posalpha,posalpha)*alpha +e(posalpha)'*alpha);
    
    %
    %  calcul de r2
    %
    
    D=diag(psc);
    A = ones(size(D));
    b=1;
    lambda=1e-10;
    verbose=0;
    [beta,r2,posbeta]=monqpCinfty(2*psc,D,A,b,lambda,verbose,x,psc,betaall);
    betaall=zeros(size(D));
    betaall(posbeta)=beta;
    pos=union(posalpha,posbeta);
    
    psaux=ps(pos,pos);
    
    SelectVariablesAux=SelectedVariables;
    r2w2dif=[];
    for i=1:length(SelectVariablesAux)
        
        
        xnon2= x(pos,SelectVariablesAux(i)); 
        xpos=x(pos,:);
   
             
        
        switch FeatSeloption.FirstOrderMethod
        case 'absgrad'
            
            [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'grad',xpos);
            dw2r2_1= - (y(pos).*alphaall(pos))'*abs(kernelderiv_1)* (y(pos).*alphaall(pos))*r2;
            dr2w2_1= ( -(betaall(pos))'*abs(kernelderiv_1)* (betaall(pos)) + (betaall(pos))'*diag(abs(kernelderiv_1)))*w2; 
            dw2r2_2= - (y(pos).*alphaall(pos))'*abs(kernelderiv_2)* (y(pos).*alphaall(pos))*r2;
            dr2w2_2= (-(betaall(pos))'*abs(kernelderiv_2)* (betaall(pos))+ (betaall(pos))'*diag(abs(kernelderiv_2)))*w2; 
            r2w2dif(i)= (dw2r2_1+dr2w2_1)^2 + (dw2r2_2+dr2w2_2)^2;
        case 'grad'
            [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'grad',xpos);
            dw2r2_1= - (y(pos).*alphaall(pos))'*(kernelderiv_1)* (y(pos).*alphaall(pos))*r2;
            dr2w2_1= (-(betaall(pos))'*(kernelderiv_1)* (betaall(pos)) + (betaall(pos))'*diag(kernelderiv_1))*w2; 
            dw2r2_2= - (y(pos).*alphaall(pos))'*(kernelderiv_2)* (y(pos).*alphaall(pos))*r2;
            dr2w2_2= (-(betaall(pos))'*(kernelderiv_2)* (betaall(pos)) + (betaall(pos))'*diag(kernelderiv_2))*w2; 
            r2w2dif(i)= (dw2r2_1+dr2w2_1)^2 + (dw2r2_2+dr2w2_2)^2;
        case 'scal'
            [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'scal',xpos);
            dw2r2= - (y(pos).*alphaall(pos))'*kernelderiv_1* (y(pos).*alphaall(pos))*r2;
            dr2w2= (-(betaall(pos))'*kernelderiv_1* (betaall(pos)) + (betaall(pos))'*diag(kernelderiv_1))*w2; 
            r2w2dif(i)= (dw2r2+dr2w2)^2 ;
        case 'absscal'
            [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'scal',xpos);
            dw2r2= - (y(pos).*alphaall(pos))'*abs(kernelderiv_1)* (y(pos).*alphaall(pos))*r2;
            dr2w2= (-(betaall(pos))'*abs(kernelderiv_1)* (betaall(pos)) + (betaall(pos))'*diag(abs(kernelderiv_1)))*w2; 
            r2w2dif(i)= (dw2r2+dr2w2)^2 ;
                    
        otherwise
            error(' Feature Selection First Order Method is undefined...');
        end;
    end
    
    
    [nointerest indiceDJ] = sort(r2w2dif);
    EliminatedVariables = [SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) EliminatedVariables];
    Values= [r2w2dif(indiceDJ(1:FeatSeloption.RemoveChunks)) Values];
    SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) = [];
    
end;

RankedVariables=[SelectedVariables EliminatedVariables ];

