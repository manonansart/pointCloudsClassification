function [RankedVariables,nbsvvec,Values,NbQP,iter]=FeatSelregmarginGD(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)

% Usage
%  
%  [RankedVariables,nbsvvec,Values,NbQP]=FeatSelregmarginGD(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)
%
%
%
%  each variable is weighted with a scaling parameter which are optimized through gradient
%  descent. After convergence, the variables are then weigthed according the magnitude of the scaling parameters
%  
%   x,y     : input data
%   c       : penalization of misclassified examples
%   kernel  : kernel type
%   kerneloption : kernel hyperparameters
%   verbose
%   FeatSeloption : structure containing FeatSeloption parameters
%           Fields           
%
%           GDitermax : stopping criterion. Maximal number of criterion
%           
%           GDthresh  : stopping criterion. stop when L2 norm of scaling vector variation is 
%                     below this threshold
%
% alain.rakoto@insa-rouen.fr
%   
%   \bibitem[Rakotomamonjy(2006)]{rakoto_featselreg}
%    A.~Rakotomamonjy.
%   \newblock Analysis of SVM regression bound for feature selection,
%   \newblock Neurocomputing 2006
% 14/03/2006 AR


%----------------------------------------------------------%
%              Testing Fields Existence                    % 
%----------------------------------------------------------%

if  isfield(FeatSeloption,'GDitermax')
    itermax=FeatSeloption.GDitermax;
else
    itermax=20;
end;
if  isfield(FeatSeloption,'GDthresh')
    thresh=FeatSeloption.GDthresh;
else
    thresh=0.01;
end;



[nbdata,nbvar]=size(x);
caux=diag((1/c)*ones(nbdata*2,1));
caux1=diag((1/c)*ones(length(y),1)); 
SelectedVariables = [1:nbvar]; %list of remaining variable

alphaall=[];
betaall=[];
nbsvvec=[];
Values=[];

NbQP=0;
iter=0;
scaling=ones(1,nbvar);
scalingold=scaling-1;
verboseaux=0;
if verbose
    fprintf('%s \t | %s  \t\t |  %s   \n','iter', 'Old', 'New');
end;
while    norm(scaling-scalingold)/norm(scaling) > thresh & iter<itermax
    
    
    
    xaux=x.*(ones(nbdata,1)*scaling);
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
    
    [alpha,bias,posalpha]=monqp(H,ee,A,b,Cinf,lambd,verboseaux,xaux,ps,alphaall);NbQP=NbQP+1;
    alphaall=zeros(length(H),1);
    alphaall(posalpha)=alpha;
    
    w2=alphaall(posalpha)'*H(posalpha,posalpha)*alphaall(posalpha);
    
    posAlphaStar=find(alphaall(1:n)>0);
    posAlpha=find(alphaall(n+1:2*n)> 0);
    newposalpha=sort([posAlphaStar;posAlpha]); 
    
    
    Bound=w2;
    
   %  keyboard
    for i=1:nbvar
        
        
       % ps est calculé en fonction du scaling factor, xi est les exemples
        % données % à la variable i
        xnon2= xaux(:,i); 
        xpos=xaux(:,:);
        [kernelderiv_1,kernelderiv_2]=featselkernelderivative(ps,xnon2,kernel,kerneloption,'scal',x);
        kernelderiv_1=kernelderiv_1/scaling(i);
        

  

        Hnon=Idif'*kernelderiv_1*Idif + caux;
        gradmarg_1= -alphaall'*Hnon* alphaall; 
        marggrad(i)=gradmarg_1;
        
        
        
    end
    marggrad=marggrad/norm(marggrad);
    
    %----------------------------------------------------------------------
    %           LINE SEARCH
    %----------------------------------------------------------------------
    step=1;
    scalingaux=scaling;
    while step > 1e-10;
        scalingaux=scaling-step*marggrad;
        xaux=x.*(ones(nbdata,1)*scalingaux);
        ps=svmkernel(xaux,kernel,kerneloption);
        
        n=size(xaux,1);
        I = eye(n);
        Idif = [I -I];
        H = Idif'*ps*Idif + caux;
        ee = [-epsilon+y ; -epsilon-y]; % [ alpha*   alpha]
        A = [-ones(1,n)  +ones(1,n) ]';
        b=0;   
        Cinf=inf;
        [alpha,bias,posalpha]=monqp(H,ee,A,b,Cinf,lambd,verboseaux,xaux,ps,alphaall);NbQP=NbQP+1;
        alphaall=zeros(length(H),1);
        alphaall(posalpha)=alpha;
        BoundTemp=alphaall(posalpha)'*H(posalpha,posalpha)*alphaall(posalpha);
        if BoundTemp > Bound
            step=step/5;
        else
            break
        end;
    end; 
    scalingold=scaling;
    scaling=scaling - step*marggrad;
    iter=iter+1;
    
    if verbose
        fprintf('%d \t\t |%2.2f  \t\t |  %2.2f   \n',iter,Bound, BoundTemp);
    end
    
end;

[ind,RankedVariables]=(sort(abs(scaling),2));
Values=(scaling(RankedVariables));
RankedVariables=fliplr(RankedVariables);
Values=fliplr(Values);
