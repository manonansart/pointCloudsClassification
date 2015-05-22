function [RankedVariables,nbsvvec,Values,NbQP]=FeatSelregalphaGDrandom(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)

% Usage
%  
%  [RankedVariables,nbsvvec,Values,NbQP]=FeatSelregalphaGDrandom(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)
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
%   span    : matrix for semiparametric learning
%   FeatSeloption : structure containing FeatSeloption parameters
%           Fields           
%
%           GDitermax : stopping criterion. Maximal number of criterion
%           
%           GDthresh  : stopping criterion. stop when L2 norm of scaling vector variation is 
%                     below this threshold
%           GDnbiterrandommax : number of random initialization (default 5)
%
% alain.rakoto@insa-rouen.fr
%   
%   \bibitem[Rakotomamonjy(2006)]{rakoto_featselreg}
%    A.~Rakotomamonjy.
%   \newblock Analysis of SVM regression bound for feature selection,
%   \newblock Neurocomputing 2006
% 26/04/2004 AR


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
if  isfield(FeatSeloption,'GDnbiterrandommax')
     nbiterrandommax=FeatSeloption.GDnbiterrandommax;
else
     nbiterrandommax=5;
end;



[nbdata,nbvar]=size(x);
caux=diag((1/c)*ones(nbdata*2,1));
caux1=diag((1/c)*ones(length(y),1)); 

BoundMax=inf;
scalingmat=2*rand(nbiterrandommax,nbvar)+0;
scalingmat(1,:)=ones(1,nbvar);
for iterrandom=1:nbiterrandommax
    scaling=scalingmat(iterrandom,:);
        SelectedVariables = [1:nbdata]; %
    alphaall=[];
    betaall=[];
    nbsvvec=[];
    Values=[];
    
    NbQP=0;
    iter=0;
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
        posAlphaStar=find(alphaall(1:n)>0);
        posAlpha=find(alphaall(n+1:2*n)> 0);
        newposalpha=sort([posAlphaStar;posAlpha]); 
        sumalpha=sum(alphaall);
        AlphaStar=alphaall(1:n);
        Alpha=alphaall(n+1:end);
        AlphaHat=AlphaStar-Alpha;
        %-------------------------------------------------------------------
        
        %---------------------------------------------%
        %  calcul de r2                               %  
        %---------------------------------------------%
        %     
        %     psc=ps+caux1;    
        %     kerneloptionr2.matrix=psc;
        %     [betaall,r2,posbeta]= r2smallestsphere([],[],kerneloptionr2);
        
        Bound=sumalpha;
        
        pos=sort([posAlphaStar;posAlpha]);
        psaux=ps(pos,pos);
        SelectVariablesAux=SelectedVariables;
        
        for i=1:nbvar
            
            
            xnon2= xaux(pos,i); 
            xpos=xaux(pos,:);
            
            [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'scal',xpos);
            kernelderiv_1=kernelderiv_1/scaling(i);
            % equation4.2 papier CJ Lin
            
            M= [psaux + caux1(pos,pos)  ones(length(pos),1);ones(1,length(pos)) 0];
            dalphahatb=M\([-kernelderiv_1*AlphaHat(pos); 0]);
            dalphahat=dalphahatb(1:end-1);
            [aux,ind,ind2]=intersect(pos,posAlphaStar);
            dAlphaStar=dalphahat(ind);    % ind are the indice of the value of posAlphaStar in pos
            % hence we are getting the derivative of the AlphaStar  
            [aux,ind]=intersect(pos,posAlpha);
            dAlpha=-dalphahat(ind);
            dsumalpha=sum(dAlphaStar)+sum(dAlpha);
            %-----------------------------------------
            
            % dw2r2=  dsumalpha*r2;
            % dr2w2= (-(betaall(pos))'*kernelderiv_1* (betaall(pos)) + (betaall(pos))'*diag(kernelderiv_1))*sumalpha; 
            
            
            alphagrad(i)= dsumalpha ;
            
        end
        alphagrad=alphagrad/norm(alphagrad);
        
        %----------------------------------------------------------------------
        %           LINE SEARCH
        %----------------------------------------------------------------------
        step=1;
        scalingaux=scaling;
        while step > 1e-10;
            scalingaux=scaling-step*alphagrad;
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
            %-------------------------------------------------------------------
            
            %---------------------------------------------%
            %  calcul de r2                               %  
            %---------------------------------------------%
            %         
            %         psc=ps+caux1;
            %         kerneloptionr2.matrix=psc;
            %         [betaall,r2,posbeta]= r2smallestsphere([],[],kerneloptionr2);NbQP=NbQP+1;
            BoundTemp=sum(alpha);
            if BoundTemp > Bound
                step=step/5;
            else
                break
            end;
        end;
        
        
        scalingold=scaling;
        scaling=scaling - step*alphagrad;
        iter=iter+1;
        
        if verbose
            fprintf('%d \t\t |%2.2f  \t\t |  %2.2f   \n',iter,Bound, BoundTemp);
        end
        
    end;
    
    if Bound < BoundMax
        [ind,RankedVariables]=(sort(abs(scaling),2));
        Values=(scaling(RankedVariables));
        RankedVariables=fliplr(RankedVariables);
        Values=fliplr(Values);
        BoundMax=Bound;
    end;
end;