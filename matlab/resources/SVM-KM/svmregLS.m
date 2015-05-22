function [xsup,ysup,w,b,newpos] = svmregLS(x,y,C,epsilon,kernel,kerneloption,lambda,verbose,qpsize)

%  USAGE 
% [xsup,ysup,w,b] = svmregLS(x,y,C,epsilon,kernel,kerneloption,lambda,verbose)
% 
% This function process the SVM regression model using a linear epsilon insensitive cost.
%
% 
% 
% INPUT
%
% Training set
%    x  : input data 
%    y  : output data
% parameters
%		C		: Bound on the lagrangian multipliers     
% 		epsilon 	 : e-tube around the solution
%		kernel		: kernel function. classical kernels are
%
%		Name			parameters
%		'poly'		polynomial degree
%		'gaussian'	gaussian standard deviation
% 
%			for more details see svmkernel
%		
% 		kerneloption : parameters of kernel
%
%
%   lambda : conditioning parameter for QP methods
%   verbose : display outputs (default value is 0: no display)



%   27/10/03 A.Rakotomamonjy Including SVMkernel


if nargin < 3
    C = 1000;
end;

if nargin < 4
    epsilon = 0.1;
end;

if nargin < 5
    kernel='gaussian';
    kerneloption = 1;
end;

if nargin < 7
    lambda = .0000001;
end;
if nargin <8
    verbose=0;
end;
if isstruct(x)
    if length(x.indice)~=length(y)
        error('Length of x and y should be equal');
    end;
end

fprintf('Large Scale \n');



KKTTol=1e-3;

% n = length(x);
% ps  =  zeros(n,n);		
% ps=svmkernel(x,kernel,kerneloption);
% K = ps;
% 
% H=[K -K;-K K];
% c = [-epsilon+y ; -epsilon-y];
% A = [ones(1,n)  -ones(1,n) ]';
% b=0;   
% 
% [alpha,bias,pos,mu]=monqp(H,c,A,b,C,lambda,verbose,x,ps);
% 
% 
% aux=zeros(length(H),1);
% aux(pos)=alpha;
% alpha=aux;
% newpos=find(alpha(1:n)>0|alpha(n+1:2*n)> 0);
% w = alpha(newpos)-alpha(n+newpos); 
% 
% 
% xsup = x(newpos,:);
% ysup = y(newpos);
% nsup =length(newpos);
% b=bias;

%-------------------------------------------
%keyboard
n = length(y);
c = [-epsilon+y ; -epsilon-y];
A = [ones(1,n)  -ones(1,n) ]';
Optimality=0;
qpsize=min(qpsize,round(0.75*n));
chunksize=qpsize;
aux=randperm(2*n);
WorkingSet=[(1:qpsize/2)'; (n+1:n+qpsize/2)'];

Alpha=zeros(n,1);
Alpha(1)=C/2;
AlphaStar=zeros(n,1);
AlphaStar(1)=C/2;
iter=0;
while ~Optimality
    
    
    % Preparing Data for solving the subproblem
    
    %  SignMatrix=((sign(1:2*n<n)-0.5)*2)'*((sign(1:2*n<n)-0.5)*2);
    Beta=[Alpha;AlphaStar];
    %Ks=svmkernel(x(WorkingSet,:),kernel,kerneloption).*SignMatrix(WorkingSet,WorkingSet);
    FixedSet=setdiff(1:2*n,WorkingSet);
    % Ksf=svmkernel(x(WorkingSet,:),kernel,kerneloption,x(FixedSet,:)).*SignMatrix(WorkingSet,FixedSet); % A OPTIMISER
    %     Ks1=H(WorkingSet,WorkingSet);
    %     Ksf1=H(WorkingSet,FixedSet);
    %   keyboard
    %     
    %    Calcul de Ks
    %    Ks=H(WorkingSet,WorkingSet);
    
    IndiceAlpha=WorkingSet(find(WorkingSet<=n));
    IndiceAlphaStar=WorkingSet(find(WorkingSet>n))-n;
    %   Ks=zeros(qpsize,qpsize);
    
    % Left Up side of Ks
    NbAlphaWorkingSet=length(IndiceAlpha);
    chunks=ceil(NbAlphaWorkingSet/chunksize);
    LU=zeros(NbAlphaWorkingSet,NbAlphaWorkingSet);
    for ch1=1:chunks
        ind1=(1+(ch1-1)*chunksize) : min( NbAlphaWorkingSet, ch1*chunksize);
        for ch2=1:chunks
            ind2=(1+(ch2-1)*chunksize) : min(NbAlphaWorkingSet, ch2*chunksize);
            
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(IndiceAlpha(ind1),:);
                x2=x(IndiceAlpha(ind2),:);
            else
                x1=fileaccess(x.datafile,x.indice(IndiceAlpha(ind1)),x.dimension);
                x2=fileaccess(x.datafile,x.indice(IndiceAlpha(ind2)),x.dimension);
                
            end;   
            
            %-----------------------------------------------------------  
            % LU(ind1,ind2)=svmkernel(x(IndiceAlpha(ind1),:),kernel,kerneloption,x(IndiceAlpha(ind2),:));
            %-----------------------------------------------------------  
            LU(ind1,ind2)=svmkernel(x1,kernel,kerneloption,x2);
        end;
    end
    % Right side of Ks
    NbAlphaStarWorkingSet=length(IndiceAlphaStar);
    chunks=ceil(NbAlphaStarWorkingSet/chunksize);
    RB=zeros(NbAlphaStarWorkingSet,NbAlphaStarWorkingSet);
    for ch1=1:chunks
        ind1=(1+(ch1-1)*chunksize) : min( NbAlphaStarWorkingSet, ch1*chunksize);
        for ch2=1:chunks
            ind2=(1+(ch2-1)*chunksize) : min(NbAlphaStarWorkingSet, ch2*chunksize);
            
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(IndiceAlphaStar(ind1),:);
                x2=x(IndiceAlphaStar(ind2),:);
            else
                x1=fileaccess(x.datafile,x.indice(IndiceAlphaStar(ind1)),x.dimension);
                x2=fileaccess(x.datafile,x.indice(IndiceAlphaStar(ind2)),x.dimension);
                
            end;  
            %-----------------------------------------------------------  
            %  RB(ind1,ind2)=svmkernel(x(IndiceAlphaStar(ind1),:),kernel,kerneloption,x(IndiceAlphaStar(ind2),:));
            %-----------------------------------------------------------  
            RB(ind1,ind2)=svmkernel(x1,kernel,kerneloption,x2);
        end;
    end
    
    % Right UP side of KS
    RU=zeros(NbAlphaWorkingSet,NbAlphaStarWorkingSet);
    chunks1=ceil(NbAlphaWorkingSet/chunksize);
    chunks2=ceil(NbAlphaStarWorkingSet/chunksize);
    for ch1=1:chunks1
        ind1=(1+(ch1-1)*chunksize) : min( NbAlphaWorkingSet, ch1*chunksize);
        for ch2=1:chunks2
            ind2=(1+(ch2-1)*chunksize) : min(NbAlphaStarWorkingSet, ch2*chunksize);
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(IndiceAlpha(ind1),:);
                x2=x(IndiceAlphaStar(ind2),:);
            else
                x1=fileaccess(x.datafile,x.indice(IndiceAlpha(ind1)),x.dimension);
                x2=fileaccess(x.datafile,x.indice(IndiceAlphaStar(ind2)),x.dimension);
                
            end;  
            %-----------------------------------------------------------     
            %RU(ind1,ind2)=svmkernel(x(IndiceAlpha(ind1),:),kernel,kerneloption,x(IndiceAlphaStar(ind2),:));
            %-----------------------------------------------------------     
            RU(ind1,ind2)=svmkernel(x1,kernel,kerneloption,x2);
        end;
    end
    
    Ks=[LU -RU; -RU' RB];
    
    %----------------------------------------------------------------------------------------------------------
    % Calcul de Ksf
    % Ksf=H(WorkingSet,FixedSet);
    
    clear LU RU RB
    
    IndiceFixedAlpha=FixedSet(find(FixedSet<=n));
    NbAlphaFixedSet=length( IndiceFixedAlpha);
    
    LU=zeros(NbAlphaWorkingSet,NbAlphaFixedSet);
    chunks1=ceil(NbAlphaWorkingSet/chunksize);
    chunks2=ceil(NbAlphaFixedSet/chunksize);
    for ch1=1:chunks1
        ind1=(1+(ch1-1)*chunksize) : min( NbAlphaWorkingSet, ch1*chunksize);
        for ch2=1:chunks2
            ind2=(1+(ch2-1)*chunksize) : min(NbAlphaFixedSet, ch2*chunksize);
            
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(IndiceAlpha(ind1),:);
                x2=x(IndiceFixedAlpha(ind2),:);
            else
                x1=fileaccess(x.datafile,x.indice(IndiceAlpha(ind1)),x.dimension);
                x2=fileaccess(x.datafile,x.indice(IndiceFixedAlpha(ind2)),x.dimension);
                
            end;  
            %-----------------------------------------------------------     
            %            LU(ind1,ind2)=svmkernel(x(IndiceAlpha(ind1),:),kernel,kerneloption,x(IndiceFixedAlpha(ind2),:));
            %-----------------------------------------------------------    
            LU(ind1,ind2)=svmkernel(x1,kernel,kerneloption,x2);
            
        end;
    end;
    
    IndiceFixedAlphaStar=FixedSet(find(FixedSet>n))-n;
    NbAlphaStarFixedSet=length( IndiceFixedAlphaStar);
    % keyboard
    RB=zeros(NbAlphaStarWorkingSet,NbAlphaStarFixedSet);
    chunks1=ceil(NbAlphaStarWorkingSet/chunksize);
    chunks2=ceil(NbAlphaStarFixedSet/chunksize);
    for ch1=1:chunks1
        ind1=(1+(ch1-1)*chunksize) : min( NbAlphaStarWorkingSet, ch1*chunksize);
        for ch2=1:chunks2
            ind2=(1+(ch2-1)*chunksize) : min(NbAlphaStarFixedSet, ch2*chunksize);
            
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(IndiceAlphaStar(ind1),:);
                x2=x(IndiceFixedAlphaStar(ind2),:);
            else
                x1=fileaccess(x.datafile,x.indice(IndiceAlphaStar(ind1)),x.dimension);
                x2=fileaccess(x.datafile,x.indice(IndiceFixedAlphaStar(ind2)),x.dimension);
                
            end;  
            %-----------------------------------------------------------     
            %    RB(ind1,ind2)=svmkernel(x(IndiceAlphaStar(ind1),:),kernel,kerneloption,x(IndiceFixedAlphaStar(ind2),:));
            %-----------------------------------------------------------    
            
            RB(ind1,ind2)=svmkernel(x1,kernel,kerneloption,x2);
        end;
    end;
    
    
    
    LB=zeros(NbAlphaStarWorkingSet,NbAlphaFixedSet);
    chunks1=ceil(NbAlphaStarWorkingSet/chunksize);
    chunks2=ceil(NbAlphaFixedSet/chunksize);
    for ch1=1:chunks1
        ind1=(1+(ch1-1)*chunksize) : min( NbAlphaStarWorkingSet, ch1*chunksize);
        for ch2=1:chunks2
            ind2=(1+(ch2-1)*chunksize) : min(NbAlphaFixedSet, ch2*chunksize);
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(IndiceAlphaStar(ind1),:);
                x2=x(IndiceFixedAlpha(ind2),:);
            else
                x1=fileaccess(x.datafile,x.indice(IndiceAlphaStar(ind1)),x.dimension);
                x2=fileaccess(x.datafile,x.indice(IndiceFixedAlpha(ind2)),x.dimension);
                
            end;  
            %-----------------------------------------------------------     
            %                LB(ind1,ind2)=svmkernel(x(IndiceAlphaStar(ind1),:),kernel,kerneloption,x(IndiceFixedAlpha(ind2),:));
            %-----------------------------------------------------------   
            
            LB(ind1,ind2)=svmkernel(x1,kernel,kerneloption,x2);
        end;
    end;
    
    
    RU=zeros(NbAlphaWorkingSet,NbAlphaStarFixedSet);
    chunks1=ceil(NbAlphaWorkingSet/chunksize);
    chunks2=ceil(NbAlphaStarFixedSet/chunksize);
    for ch1=1:chunks1
        ind1=(1+(ch1-1)*chunksize) : min( NbAlphaWorkingSet, ch1*chunksize);
        for ch2=1:chunks2
            ind2=(1+(ch2-1)*chunksize) : min(NbAlphaStarFixedSet, ch2*chunksize);
            
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(IndiceAlpha(ind1),:);
                x2=x(IndiceFixedAlphaStar(ind2),:);
            else
                x1=fileaccess(x.datafile,x.indice(IndiceAlpha(ind1)),x.dimension);
                x2=fileaccess(x.datafile,x.indice(IndiceFixedAlphaStar(ind2)),x.dimension);
                
            end;  
            %-----------------------------------------------------------     
            %  RU(ind1,ind2)=svmkernel(x(IndiceAlpha(ind1),:),kernel,kerneloption,x(IndiceFixedAlphaStar(ind2),:));
            %---------------------
            RU(ind1,ind2)=svmkernel(x1,kernel,kerneloption,x2);
        end;
    end;
    Ksf=[ LU -RU;-LB RB];
    %     if max(max(abs(Ksf-Ksf1)))>1e-3 | max(max(abs(Ks-Ks)))>1e-3 
    %         keyboard
    %     end;
    
    %----------------------------------------------------------------------------------------------------------
    %  Préparation du sous-problème et résolution
    %
    
    %keyboard
    cs=c(WorkingSet)-Ksf*(Beta(FixedSet));
    As=A(WorkingSet,:);
    b=-A(FixedSet,:)'*Beta(FixedSet);

    
    verbose=0;
    [BetaS,bias,pos,mu]=monqp(Ks,cs,As,b,C,lambda,verbose);
    BetaF=zeros(length(WorkingSet),1);
    BetaF(pos)=BetaS;
    Beta(WorkingSet)=BetaF;
    
    % derivative
    
    % DerivAlpha=K(:,newpos)*w - y + epsilon*ones(n,1)   % ok... this is the derivative extract from the original 
    % DerivAlphaStar=-K(:,newpos)*w +y + epsilon*ones(n,1)   % problem formulation
    
    
    %Alpha=alpha(1:n);
    %AlphaStar=alpha(n+1:2*n);
    Alpha=Beta(1:n);
    AlphaStar=Beta(n+1:2*n);
    
    IndPosSVAlpha=find( Alpha>0 & Alpha <C) ;
    IndPosSVAlphaStar=find( AlphaStar>0 & AlphaStar <C) ;
    NbPosSVAlpha=length(IndPosSVAlpha);
    NbPosSVAlphaStar=length(IndPosSVAlphaStar);
    IndAlphaZero=find(Alpha==0);
    IndAlphaStarZero=find(AlphaStar==0);
    IndAlphaC=find(Alpha==C);
    IndAlphaStarC=find(AlphaStar==C);
    
    
    %
    % calcul de la derivée par boucle 
    %
    AlphaMinusAlphaStar= Alpha - AlphaStar ; 
    
    %  Obj= 1/2 *AlphaMinusAlphaStar'*K*AlphaMinusAlphaStar - AlphaMinusAlphaStar'*y + epsilon*sum(Alpha+AlphaStar);
    Obj=0; 
    chunks=ceil(n/chunksize);
    s=zeros(n,1);
    for ch1=1:chunks
        ind1=(1+(ch1-1)*chunksize) : min( n, ch1*chunksize);
        for ch2=1:chunks
            ind2=(1+(ch2-1)*chunksize) : min(n, ch2*chunksize);
            
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(ind1,:);
                x2=x(ind2,:);
            else
                x1=fileaccess(x.datafile,x.indice(ind1),x.dimension);
                x2=fileaccess(x.datafile,x.indice(ind2),x.dimension);
                
            end;   
            kchunk=svmkernel(x1,kernel,kerneloption,x2);
            %-----------------------------------------------------------  
            %kchunk=svmkernel(x(ind1,:),kernel,kerneloption,x(ind2,:));
            %-----------------------------------------------------------  
            
            s(ind1)=s(ind1)+ kchunk*AlphaMinusAlphaStar(ind2) ;
        end;
    end
    DerivAlpha=s-y+epsilon*ones(n,1);
    DerivAlphaStar=-s+y+epsilon*ones(n,1); 
    
    LambdaLowAlpha=zeros(n,1);
    LambdaLowAlphaStar=zeros(n,1);
    LambdaHighAlpha=zeros(n,1);
    LambdaHighAlphaStar=zeros(n,1);
    
    if NbPosSVAlpha+NbPosSVAlphaStar~=0
        LambdaEq=(sum(DerivAlphaStar(IndPosSVAlphaStar))- sum(DerivAlpha(IndPosSVAlpha)))/(NbPosSVAlpha+NbPosSVAlphaStar);
    else
        LambdaEq=0;
    end;
    
    
    LambdaLowAlpha(IndAlphaZero)=DerivAlpha(IndAlphaZero)+LambdaEq;
    LambdaLowAlphaStar(IndAlphaStarZero)=DerivAlphaStar(IndAlphaStarZero)-LambdaEq;
    LambdaHighAlpha(IndAlphaC)=-DerivAlpha(IndAlphaC)-LambdaEq;
    LambdaHighAlphaStar(IndAlphaStarC)=-DerivAlphaStar(IndAlphaStarC)+LambdaEq;
    
    
    Deriv=[DerivAlpha;DerivAlphaStar];
    LambdaEqVect=LambdaEq*[ones(n,1);-ones(n,1)];
    LambdaLow=[LambdaLowAlpha;LambdaLowAlphaStar] ;
    LambdaHigh=[LambdaHighAlpha;LambdaHighAlphaStar];
    KKT1=Deriv+LambdaEqVect-LambdaLow+LambdaHigh;
    Optimality=max(abs(KKT1))<KKTTol & min(LambdaLow>-KKTTol) & min(LambdaHigh>-KKTTol);
    OldWorkingSet=WorkingSet;
    if ~Optimality % Select New WorkingSet
        
        indNonKKT1=find(abs(KKT1)>KKTTol);
        indNonKKT2=find(LambdaLow<-KKTTol);
        indNonKKT3=find(LambdaHigh<-KKTTol);
        indNonKKT=unique([indNonKKT1;indNonKKT2;indNonKKT3]);
        indNonKKTAlpha=indNonKKT(find(indNonKKT <=n));
        indNonKKTAlphaStar=indNonKKT(find(indNonKKT >=n+1));
        %   %      keyboard
        NbNonKKTAlpha=length(indNonKKTAlpha);
        NbNonKKTAlphaStar=length(indNonKKTAlphaStar);
        NbNewAlphaInWorkingSet=min(NbNonKKTAlpha,qpsize/2);
        NbNewAlphaStarInWorkingSet=min(NbNonKKTAlphaStar,qpsize-NbNewAlphaInWorkingSet);
        aux1=randperm(NbNonKKTAlpha);
        aux2=randperm(NbNonKKTAlphaStar);
        WorkingSet=[indNonKKTAlpha(aux1(1:NbNewAlphaInWorkingSet)); indNonKKTAlphaStar(aux2(1: NbNewAlphaStarInWorkingSet))];
        if length(WorkingSet) < qpsize
            FixedSet=setdiff(1:2*n,WorkingSet);
            aux=randperm(length(FixedSet));
            
            WorkingSet=[WorkingSet;FixedSet(aux(1:qpsize-length(WorkingSet)))'];
        end;
        
        WorkingSet=sort(WorkingSet);
        ChangedAlpha=length(setdiff(WorkingSet,OldWorkingSet));
        % 
    end;
    iter=iter+1;
    fprintf('i : %d Changed Alpha:%d Nb KKT Violation: %d Objective Val: %2.5f\n', iter,ChangedAlpha,NbNonKKTAlpha+NbNonKKTAlphaStar,Obj);
    
    
end;
fprintf('sortie\n');


% it is over

newpos=find(Alpha>0|AlphaStar> 0);
w = Alpha(newpos)-AlphaStar(newpos); 

if ~isfield(x,'datafile')
xsup = x(newpos,:);
else
    xsup=x;
    xsup.indice=x.indice(newpos);
end;
ysup = y(newpos);
nsup =length(newpos);
b=LambdaEq;