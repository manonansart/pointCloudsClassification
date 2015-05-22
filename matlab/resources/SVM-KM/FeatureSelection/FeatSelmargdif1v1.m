function [RankedVariables,nbsvvec,Values]=FeatSelmargdif1v1(x,y,nbclass,c,kernel,kerneloption,verbose,span,FeatSeloption)

% Usage
%  
%   [RankedVariables,nbsvvec,Values]=FeatSelmargdif1v1(x,y,nbclass,c,kernel,kerneloption,verbose,span,FeatSeloption)
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
%           AlphaApprox : O for retraining,  1 for approximation 
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
%----------------------------------------------------------%
%              Initialization                 % 
%----------------------------------------------------------%
SelectedVariables = [1:size(x,2)]; %list of remaining variable
EliminatedVariables = []; %list of eliminated variables
Values=[];

alphaall=[];
betaall=[];
nbsvvec=[];
yaux=y;
while length(SelectedVariables)~=0
    if verbose==0
        fprintf('.');
    end;
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
    margdifall=zeros(1,length(SelectedVariables));
    for k=1:nbclass-1
        for j=i+1:nbclass
            
            %-------------------------Consider Only the appropriate data---------------------------------------
            % this can be done in the outer loop 
            indi=find(yaux==k);
            indj=find(yaux==j);
            xaux=[x(indi,SelectedVariables); x(indj,SelectedVariables)];
            y=[ones(length(indi),1);-ones(length(indj),1)];
            
            %----------------------------------------------------------------------------------------------
            caux=diag((1/c)*ones(length(y),1));
            ps=svmkernel(xaux,kernel,kerneloption);
            lambda=1e-7;
            psc=ps+caux;
            H =psc.*(y*y'); 
            e = ones(size(y));
            A = y;            b = 0;
            
        %    keyboard
            %-------- This is a QP algorithm that should be replaced by your own QP ---------
            [alpha , lambda , pos] =  monqpCinfty(H,e,A,b,lambda,verbose,x,psc); 
            %--------------------------------------------------------------------------------
            
            nbsv=length(pos);
            nbsvvec=[nbsv nbsvvec];
            alphaall=zeros(size(e));
            alphaall(pos)=alpha;
            w2=2*(-0.5*alpha'*H(pos,pos)*alpha +e(pos)'*alpha);
            
            SelectVariablesAux=SelectedVariables;
            margdif=[];
  
            psaux=ps(pos,pos);  
            SelectVariablesAux=SelectedVariables;
            for i=1:length(SelectedVariables)
                xnon2= x(pos,SelectVariablesAux(i)); 
                xpos=x(pos,:);
                switch FeatSeloption.FirstOrderMethod
                    
                case 'absgrad'    
                    
                    [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'grad',xpos);
                    gradmarg_1= -(y(pos).*alphaall(pos))'*abs(kernelderiv_1)* (y(pos).*alphaall(pos));
                    gradmarg_2= -(y(pos).*alphaall(pos))'*abs(kernelderiv_2)* (y(pos).*alphaall(pos));
                    margdif(i)=gradmarg_1.^2+gradmarg_2.^2;
                case 'grad'    
                    [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'grad',xpos);
                    gradmarg_1= -(y(pos).*alphaall(pos))'*kernelderiv_1* (y(pos).*alphaall(pos));
                    gradmarg_2= -(y(pos).*alphaall(pos))'*kernelderiv_2* (y(pos).*alphaall(pos));
                    margdif(i)=gradmarg_1.^2+gradmarg_2.^2;
                    
                case 'scal'
                    [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'scal',xpos);
                    gradmarg_1= -(y(pos).*alphaall(pos))'*kernelderiv_1* (y(pos).*alphaall(pos));
                    margdif(i)=gradmarg_1; % on fait la somme apres
              
                    
                case 'absscal'
                    [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'scal',xpos);
                    gradmarg_1= -(y(pos).*alphaall(pos))'*abs(kernelderiv_1)* (y(pos).*alphaall(pos));
                    margdif(i)=gradmarg_1.^2;
                    
                otherwise
                    error(' Feature Selection First Order Method is undefined...');
                end;
                  margdifall(i)=margdifall(i)+margdif(i);
            end
            
            %------------------------------------------------------------------------------------------------------------------
          
            
        end
    end;
    
    
    
    
    [nointerest indiceDJ] = sort(abs(margdifall));
%    [nointerest indiceDJ] = sort(margdifall);
    EliminatedVariables = [SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) EliminatedVariables];
    Values= [margdifall(indiceDJ(1:FeatSeloption.RemoveChunks)) Values];
    SelectedVariables(indiceDJ(1:FeatSeloption.RemoveChunks)) = [];
    
end;

RankedVariables=[ EliminatedVariables ];

