function [RankedVariables,nbsvvec,Values,NbQP,NbInv,iter]=FeatSelspanboundGD(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)

%   USAGE
%
%  [RankedVariables,nbsvvec,Values]=FeatSelspanboundGD(x,y,c,epsilon,kernel,kerneloption,verbose,FeatSeloption)
%
%   Backward Variable Selection using grad of Span Estimate as a criterion   
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
%           SEeta     : regularisation term of span estimate. see CJ Lin paper
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

if  isfield(FeatSeloption,'SEeta')
    eta=FeatSeloption.SEeta;
else
    eta=0.001;
end;

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



% Initialization
[nbdata,nbvar]=size(x);
lambd=1e-8;
lambdaregul=1e-8;
caux=diag((1/c)*ones(nbdata*2,1));
caux1=diag((1/c)*ones(nbdata,1));
scaling=ones(1,nbvar);
SelectedVariables = [1:nbdata]; %

alphaall=[];
nbsvvec=[];
Values=[];
iter=0;
NbQP=0;
NbInv=0;
scalingold=scaling-1;
 verboseaux=0;
 
 if verbose
     fprintf('%s \t | %s  \t\t\t |  %s   \n','iter', 'Old', 'New');
 end;
 
 
while    norm(scaling-scalingold)/norm(scaling) > thresh & iter<itermax
    
    
    xaux=x.*(ones(nbdata,1)*scaling);
    ps=svmkernel(xaux,kernel,kerneloption);
    lambd=1e-7;
    n=size(xaux,1);
    I = eye(n);
    Idif = [I -I];
    H = Idif'*ps*Idif + caux;
    ee = [-epsilon+y ; -epsilon-y]; % [ alpha*   alpha]
    A = [-ones(1,n)  +ones(1,n) ]';
    b=0;   
    Cinf=inf;
    [alphatemp,bias,posalpha]=monqp(H,ee,A,b,Cinf,lambd,verboseaux,x,ps,alphaall);NbQP=NbQP+1;
    nbsv=length(alphatemp); 
    nbsvvec=[nbsvvec nbsv];
    alphaall=zeros(length(H),1);
    alphaall(posalpha)=alphatemp;
    
    % calcul de Alpha et AlphaStar
    AlphaStar=alphaall(1:n);
    Alpha=alphaall(n+1:end);
    AlphaHat=AlphaStar-Alpha;
    AlphaPlusAlphaStar=Alpha+AlphaStar;
    posAlphaStar=find(alphaall(1:n)>0);
    posAlpha=find(alphaall(n+1:2*n)> 0);
    
    newpos=sort([posAlphaStar;posAlpha]); 
    
    
    % Calcul du span bound
  
    D=(eta./(Alpha(newpos)+AlphaStar(newpos)));
    ktilde=ps(newpos,newpos) +caux1(newpos,newpos);
    M=[ktilde ones(nbsv,1); ones(1,nbsv) 0];
    Mtilde=M+diag([D;0]);
    invMtilde=inv(Mtilde+lambdaregul*eye(nbsv+1)); NbInv=NbInv+1;
    MtildeInvDiag=1./diag(invMtilde);
    sp2=MtildeInvDiag(1:nbsv)-D;
    
    Bound=(AlphaPlusAlphaStar(newpos))'*sp2;

    psaux=ps(newpos,newpos);
    
    SelectVariablesAux=SelectedVariables;
    T=[];
    
    for i=1:nbvar
        
        
        xnon2= xaux(newpos,i); 
        xpos=xaux(newpos,:);
        [kernelderiv_1,kernelderiv_2]=featselkernelderivative(psaux,xnon2,kernel,kerneloption,'scal',xpos);
         kernelderiv_1=kernelderiv_1/scaling(i);
        
        % Calcul de dAlpha et dAlphaStar
        M= [psaux + caux1(newpos,newpos)  ones(length(newpos),1);ones(1,length(newpos)) 0];
        dalphahatb=M\([-kernelderiv_1*AlphaHat(newpos); 0]);NbInv=NbInv+1;
        dalphahat=dalphahatb(1:end-1);
        
        % ind are the indice of the value of posAlphaStar in pos
        % hence we are getting the derivative of the AlphaStar  
        [aux,ind]=intersect(newpos,posAlphaStar);
        dAlphaStar=dalphahat(ind);    
         [aux,ind2]=intersect(newpos,posAlpha);
        dAlpha=-dalphahat(ind2);
        dAlphaPlusAlphaStar=zeros(nbsv,1);
        dAlphaPlusAlphaStar(ind)=dAlphaStar;
        dAlphaPlusAlphaStar(ind2)=dAlpha;
        
        %--------------------------------------------------------
        dD=-(eta./(AlphaPlusAlphaStar(newpos).^2)).*(   dAlphaPlusAlphaStar   ); 
        %---------------------------------------------------------
        %
        dMtilde= [kernelderiv_1+diag(dD) zeros(nbsv,1); zeros(1,nbsv) 0];
        dMtildeinvDiag= diag(invMtilde*dMtilde*invMtilde);
        dSp= - (MtildeInvDiag(1:nbsv)).^2.*dMtildeinvDiag(1:nbsv)- dD;
        
        
        
        
        T(i)=  ((dAlphaPlusAlphaStar'*sp2)+  (AlphaPlusAlphaStar(newpos))'*dSp);  
        
        
    end
    
 %   keyboard
    T=T/norm(T);
    
    %----------------------------------------------------------------------
    %           LINE SEARCH
    %----------------------------------------------------------------------
    step=1;
    scalingaux=scaling;
    while step > 1e-10;
        scalingaux=scaling-step*T;
        xaux=x.*(ones(nbdata,1)*scalingaux);
        ps=svmkernel(xaux,kernel,kerneloption);
        lambd=1e-7;
        n=size(xaux,1);
        I = eye(n);
        Idif = [I -I];
        H = Idif'*ps*Idif + caux;
       
        [alphatemp,bias,posalpha]=monqp(H,ee,A,b,Cinf,lambd,verboseaux,x,ps,alphaall);NbQP=NbQP+1;
        nbsv=length(alphatemp); 
      
        alphaall=zeros(length(H),1);
        alphaall(posalpha)=alphatemp;
        AlphaStar=alphaall(1:n);
        Alpha=alphaall(n+1:end);
        AlphaHat=AlphaStar-Alpha;
        AlphaPlusAlphaStar=Alpha+AlphaStar;
        posAlphaStar=find(alphaall(1:n)>0);
        posAlpha=find(alphaall(n+1:2*n)> 0);
        
        newpos=sort([posAlphaStar;posAlpha]); 
        
        D=(eta./(Alpha(newpos)+AlphaStar(newpos)));
        ktilde=ps(newpos,newpos) +caux1(newpos,newpos);
        M=[ktilde ones(nbsv,1); ones(1,nbsv) 0];
        Mtilde=M+diag([D;0]);
        invMtilde=inv(Mtilde+lambdaregul*eye(nbsv+1)); NbInv=NbInv+1;
        MtildeInvDiag=1./diag(invMtilde);
        sp2=MtildeInvDiag(1:nbsv)-D;
        
        BoundTemp=(AlphaPlusAlphaStar(newpos))'*sp2;
        if BoundTemp > Bound
            step=step/5;
        else
            break
        end;
    end; 
    scalingold=scaling;
    scaling=scaling -step*T;
    iter=iter+1;  
    if verbose
        fprintf('%d \t\t |%2.2f  \t\t |  %2.2f   \n',iter,Bound, BoundTemp);
    end

    
end;

[ind,RankedVariables]=(sort(abs(scaling),2));
Values=(scaling(RankedVariables));
RankedVariables=fliplr(RankedVariables);
Values=fliplr(Values);


