function [alphamat,alpha0vec,lambdavec,event]=regpathsvmoneclass(xapp,kernel,kerneloption,verbose,options);


nuinit=0.999;
lambdaseuil=1.1;
lambdaseuil=size(xapp,1)*0.01;
epsilon=1e-8;
DOQP=1;
lambd=1e-8;
chouia=1e-3;


event=[];
lambdavec=[];
alphamat=[];
alpha0vec=[];

%--------------------------------------
% initialisation
%--------------------------------------
[nbtrain,dim]=size(xapp);
lambda=nuinit*nbtrain;
Kapp=normalizekernel(xapp,kernel,kerneloption) + epsilon*eye(nbtrain);


%
% INITIALIZATION
if DOQP
    c=zeros(nbtrain,1);
    A=ones(nbtrain,1);
    b=lambda;
    C= 1;
    
    indrand=randperm(nbtrain);
    Elambda=floor(lambda);
    alphainit(indrand(1:Elambda),1)=1;alphainit(indrand(Elambda+1),1)=lambda-Elambda;
    [alphaaux, multiplier, pos]=monqp(Kapp/lambda,c,A,b,C,lambd,verbose,[],[],alphainit);
    if sum(alphaaux>1 | alphaaux < 0 )>0
        fprintf('Error Init')
    end;
    alpha=zeros(nbtrain,1);
    alpha(pos)=alphaaux;
    alpha0=-lambda*multiplier;
    fx=(Kapp*alpha-alpha0)/lambda;
    lambdavec=[lambdavec lambda];
    alphamat=[alphamat alpha];
    alpha0vec=[alpha0vec alpha0];
    elbow=find(abs(fx)<lambd);
    left=find(fx<-lambd);
    right=find(fx>lambd);
else
    %--------------------------------------
    %% process fx for the parzen windows
    alpha=ones(nbtrain,1);
    alpha0=sqrt(alpha'*Kapp*alpha);
    fx=(Kapp*alpha-alpha0)/nbtrain; 
    
    indaux=find(fx<0);
    [aux,indaux1]=max(fx(indaux));
    indaux1=find(fx==aux); % see if there is any doublons
    if length(indaux1)>1
        warning('Doublons');
        indaux1=indaux1(1);
    end;
    alpha(indaux(indaux1))=lambda-floor(lambda);
    alpha0=Kapp(indaux(indaux1(1)),:)*alpha;
    fx=(Kapp*alpha-alpha0)/lambda;
    lambdavec=[lambdavec lambda];
    alphamat=[alphamat alpha];
    alpha0vec=[alpha0vec alpha0];
    elbow=[indaux(indaux1)];
    right=[];
    left=setdiff(1:nbtrain,elbow)';
end;


%
%           Main algorithm
%
while lambda > lambdaseuil & lambda > 1
    if verbose ==1
        fprintf('\rlambda=%3.3f',lambda);
    end;
    
    nbelbow=length(elbow);
    if nbelbow ~=0
        Una=[zeros(nbelbow,1);1];
        A=[Kapp(elbow,elbow) -ones(nbelbow,1); ones(1,nbelbow) 0];
        ba=A\Una;
        balpha=ba(1:end-1);
        bo=ba(end);
        
        % case elbow to 0 or 1
        lambda1=lambda + (ones(nbelbow,1)-alpha(elbow))./balpha;
        lambda2=lambda + (zeros(nbelbow,1)-alpha(elbow))./balpha;
        %case L and R
        %hellxi=sum(Kapp(:,elbow).*(ones(nbtrain,1)*(balpha)'),2) - bo;
        hellxi=Kapp(:,elbow)*balpha - bo;
        
        lambda3=lambda*(fx(left)-hellxi(left))./(-hellxi(left));
        lambda4=lambda*(fx(right)-hellxi(right))./(-hellxi(right));
        ind1=find( (lambda1+epsilon)<lambda);
        ind2=find((lambda2+epsilon)<lambda);
        ind3=find((lambda3+epsilon)<lambda);
        ind4=find((lambda4+epsilon)<lambda);     
        
        [lambdanew]=max([lambda1(ind1);lambda2(ind2);lambda3(ind3);lambda4(ind4)]);
        if isempty(lambdanew)
                fprintf('Exit due to no new lambda ... \n');
                break    
        end;
        fx=lambda/lambdanew*(fx-hellxi)+hellxi;
        alpha(elbow)=alpha(elbow)-(lambda-lambdanew)*balpha;
        alpha0=alpha0-(lambda-lambdanew)*bo;
        lambda=lambdanew;
        
        % mise à jour des ensembles
        if ismember(lambdanew,lambda1)% event 1 elbow to left 
            ind=find(abs(lambda1-lambdanew)<epsilon);
            left=[left;elbow(ind)];
            elbow(ind)=[];
            event=[event;1];
        end;
        if ismember(lambdanew,lambda2)% event 2 elbow to right 
            ind=find(abs(lambda2-lambdanew)<epsilon);
            right=[right;elbow(ind)];
            elbow(ind)=[];
            event=[event;2];
        end;
        if ismember(lambdanew,lambda3)% event 3 left to elbow 
            ind=find(abs(lambda3-lambdanew)<epsilon);
            elbow=[elbow;left(ind)];
            left(ind)=[];
            event=[event;3];
        end;
        if ismember(lambdanew,lambda4)% event 4 rigth to elbow 
            ind=find(abs(lambda4-lambdanew)<epsilon);
            elbow=[elbow;right(ind)];
            right(ind)=[];
            event=[event;4];
        end;
        
        
    else
        % if elbow is empty then redo initialisation
        
        lambda=lambda-chouia;
        
        %--------------------------------------------------
        % 
        %  initialisation with QP
        if DOQP
            c=zeros(nbtrain,1);
            A=ones(nbtrain,1);
            b=lambda;
            C= 1;
            alphainit=alpha;
            indaux=find(fx<0);
            [aux,indaux1]=max(fx(indaux));
            alphainit(indaux(indaux1))=1-chouia;
            
            [alphaaux, multiplier, pos]=monqp(Kapp/lambda,c,A,b,C,epsilon,verbose,[],[],alphainit);
            if sum(alphaaux>1)>0
                fprintf('*')
                break
            end;
            alpha=zeros(nbtrain,1);
            alpha(pos)=alphaaux;
            alpha0=-lambda*multiplier;
            fx=(Kapp*alpha-alpha0)/lambda;
            elbow=find(abs(fx)<epsilon);
            left=find(fx<-epsilon);
            right=find(fx>epsilon);
        else
            % alternative heuristics
            alphainit=alpha;
            indaux=find(fx<0);
            [aux,indaux1]=max(fx(indaux));
            indaux1=find(fx(indaux)==aux); % see if there is any doublons
            if length(indaux1)>1
                warning('doublons');
                indaux1=indaux1(1);
            end;
            
            alphainit(indaux(indaux1))=1-chouia;
            alpha=alphainit;
            alpha0=Kapp(indaux(indaux1),:)*alpha;
            if isempty(alpha0)
                alpha0=0;
            end;
            fx=(Kapp*alpha-alpha0)/lambda;
            elbow=[elbow;indaux(indaux1)];
            elbow=find(abs(fx)<epsilon);
            left=find(fx<-epsilon);
            right=find(fx>epsilon);
            event=[event;5];
        end;
        
    end;   
    alpha(right)=0;
    alpha(left)=1;
    indaux=find(fx(right)<epsilon);
    fx(right(indaux))=epsilon;
    
    lambdavec=[lambdavec lambda];
    alphamat=[alphamat alpha];
    alpha0vec=[alpha0vec alpha0];
    
%     elbow=sort(elbow);
%     right=sort(right);
%     left=sort(left);
    
    
   if verbose>2
    
        xsup=xapp;
        w=alpha;
        w0=-alpha0;
        ypred = svmval(xtest,xsup,w,w0,kernel,kerneloption,1)/lambda;
        ypred=reshape(ypred,nn,nn);
        figure(1);  
        subplot(2,1,1)
        %contourf(xtest1,xtest2,ypred,50);shading flat;
        hold on
        [cc,hh]=contour(xtest1,xtest2,ypred,[0 0],'k');
        % clabel(cc,hh)
        set(hh,'LineWidth',2);
        h1=plot(xapp(:,1),xapp(:,2),'+r'); 
       
        set(h1,'LineWidth',2);
         h1=plot(xapp(elbow,1),xapp(elbow,2),'dg'); 
        subplot(2,1,2);
        plot(alphamat');
        
        drawnow
         end;
    
end

%plot(alphamat');

