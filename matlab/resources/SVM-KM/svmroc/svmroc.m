function [xsup,w,w0,pos,timeps,alpha,matriceind]=svmroc(x,y,C,kppv,margin,lambda,kernel,kerneloption,verbose,span,qpsize,kkttol,matriceind,alphainit);

% USAGE
% 
% [xsup,w,w0,pos,timeps,alpha,matriceind]=svmroc(x,y,C,kppv,margin,lambda,kernel,kerneloption,verbose,span,qpsize,matriceind)
%
%
% SVM ROC Optimizer that can handle LS problem and large neighboorhood. This
% algorithm uses a decomposition procedure.
%
%  x y          the learning data and labels
%  C            penalization parameters
%  kppv         the number of neighboor to consider. choose kppv=inf for
%               genuine ROC-SVM with no approx.
%  margin       the margin for ranking
%  lambda       conditioning parameter for the qp problem e.g 1e-7
%  kernel       the kernel type e.g 'gaussian' or 'poly'
%  kerneloption kernel parameters
%  verbose      verbosity of the algo 0 or 1
%  span         type of semi parametric function e.g 1
%  qpsize       size of qp algorithm
%
%
% Outputs as usual for SVM except that xsup is a cell containing
% the couple of positive and negative support vectors.
% 
% see also svmrocval
%
% 

% 30/07/2004 A. Rakotomamonjy

if nargin< 14
    alphainit=[];
end;
if nargin< 13
    matriceind=[];
end;
if nargin< 12
    kkttol=1e-4;
end;

if size(alphainit,1)~=size(matriceind,1)
    error('matriceind and alphainit initialization size mismatch...');
end;

iteration=0;
difftol=1e-7;
chunksize=qpsize;

indpos=find(y==1);
indneg=find(y==-1);
nbpos=length(indpos);
nbneg=length(indneg);

timeps=0;


%-------------------------------------------------------------------------
%
% dist in feature space of each positive example to negative examples
%
% the idea here is to select a subset of couple for
% optimizing the ranking
%
% 
% CASE 1 : select only the k-nearest negative neighbor of positive examples
if isempty(matriceind);
    if kppv~=inf
        for i=1:nbpos+nbneg
            norme2(i)=svmkernel(x(i,:),kernel,kerneloption);
        end;
        
        matriceindneg=[];
        % select only the k-nearest positive neighbor of negative xamples
        for i=1:nbneg
            aux1=svmkernel(x(indneg(i),:),kernel,kerneloption,x(indpos,:));
            dist=norme2(indneg(i))*ones(1,nbpos) + norme2(indpos) - 2*aux1 ;
            [aux,indicesorted]=sort(dist');
            minim=min(length(indicesorted),kppv);
            matriceindneg=[matriceindneg; i*ones(minim,1) indicesorted(1:minim)];
        end;
        vect=unique(matriceindneg(:,2)); 
        % process only these couples of nn of these positive samples.
        dist=[];
        matriceind=[];
        for i=1:length(vect)
            aux1=svmkernel(x(indpos(vect(i)),:),kernel,kerneloption,x(indneg,:));
            dist=norme2(indpos(vect(i)))*ones(1,nbneg) + norme2(indneg) - 2*aux1 ;
            [aux,indicesorted]=sort(dist');
            minim=min(length(indicesorted),kppv);
            matriceind=[matriceind; vect(i)*ones(minim,1) indicesorted(1:minim)];
        end;
        
    else   % Select all couples
        %         k=1;
        %         for i=1:nbneg
        %             for j=1:nbpos
        %                 matriceind(k,:)=[j i];
        %                 k=k+1;    
        %             end;
        %         end;
        
        [aux1 aux2]  = meshgrid(1:nbpos,1:nbneg); 
        [nn1,nn2]= size(aux1); 
        matriceind = [reshape(aux1 ,nn1*nn2,1) reshape(aux2 ,nn1*nn2,1)]; 
        
    end;
end;
taille=length(matriceind);
%--------Matrice stocke la liste des couples de points à traiter

alpha=zeros(taille,1);

%--------CALCUL de f(x_i)- f(x_j) pour tout les couples de matricind

if isempty(alphainit)
    alpha(1:10)=C/2;
else
    alpha=alphainit.*(alphainit<C) + C*(alphainit>=C);
end;
if taille <=qpsize;
    qpsize=taille;
end;
while 1
    SVbound=(alpha>=C-difftol);
    SV=(alpha >difftol);
    SVnonbound= (~SVbound & SV);
    
    
    indSV=find(SV);
    
    
    
%     chunks1=ceil(taille/chunksize);
%     chunks2=ceil(length(indSV)/chunksize);
%     s=zeros(taille,1);
%     for ch1=1:chunks1
%         ind1=(1+(ch1-1)*chunksize) : min( taille, ch1*chunksize);
%         for ch2=1:chunks2
%             ind2=(1+(ch2-1)*chunksize) : min(length(indSV), ch2*chunksize);
%             kchunk1=svmkernel(x(indpos(matriceind(ind1,1)),:),kernel,kerneloption,x(indpos(matriceind(indSV(ind2),1)),:))-...
%                 svmkernel(x(indpos(matriceind(ind1,1)),:),kernel,kerneloption,x(indneg(matriceind(indSV(ind2),2)),:));
%             
%             kchunk2=svmkernel(x(indneg(matriceind(ind1,2)),:),kernel,kerneloption,x(indpos(matriceind(indSV(ind2),1)),:))-...
%                 svmkernel(x(indneg(matriceind(ind1,2)),:),kernel,kerneloption,x(indneg(matriceind(indSV(ind2),2)),:));
%             
%             
%             s(ind1)=s(ind1)+( kchunk1-kchunk2)*alpha(indSV(ind2)); % (xi+ - xj-, xk+ - xl-)
%         end;
%     end
%     
    % process score positive
    chunks1=ceil(nbpos/chunksize);
    chunks2=ceil(length(indSV)/chunksize);
    spos=zeros(nbpos,1);
    for ch1=1:chunks1
        ind1=(1+(ch1-1)*chunksize) : min( nbpos, ch1*chunksize);
        for ch2=1:chunks2
            ind2=(1+(ch2-1)*chunksize) : min(length(indSV), ch2*chunksize);
            kchunk1=svmkernel(x(indpos(ind1),:),kernel,kerneloption,x(indpos(matriceind(indSV(ind2),1)),:))-...
                svmkernel(x(indpos(ind1),:),kernel,kerneloption,x(indneg(matriceind(indSV(ind2),2)),:));
            spos(ind1)=spos(ind1)+( kchunk1)*alpha(indSV(ind2)); % (xi+ - xj-, xk+ - xl-)
        end;
    end
    % process score negative
    chunks1=ceil(nbneg/chunksize);
    chunks2=ceil(length(indSV)/chunksize);
    sneg=zeros(nbneg,1);
    for ch1=1:chunks1
        ind1=(1+(ch1-1)*chunksize) : min( nbneg, ch1*chunksize);
        for ch2=1:chunks2
            ind2=(1+(ch2-1)*chunksize) : min(length(indSV), ch2*chunksize);
            kchunk1=svmkernel(x(indneg(ind1),:),kernel,kerneloption,x(indpos(matriceind(indSV(ind2),1)),:))-...
                svmkernel(x(indneg(ind1),:),kernel,kerneloption,x(indneg(matriceind(indSV(ind2),2)),:));
            sneg(ind1)=sneg(ind1)+( kchunk1)*alpha(indSV(ind2)); % (xi+ - xj-, xk+ - xl-)
        end;
    end
    
    s=reshape((spos*ones(1,nbneg)-ones(nbpos,1)*sneg')',nbpos*nbneg,1);
    
    %keyboard
    
    
    %ypred = svmrocval(xapp,xsup,w,w0,kernel,kerneloption,span); 
    % verification des contraintes de KKT
    kkt=s-margin;
    kktviolation=   (SVnonbound & ( abs(kkt)>kkttol) )|( ~SV & (kkt < -kkttol)) | ( SVbound & (kkt > kkttol));
    
    indkktviolation=find(kktviolation);
    nbkktviolation=length(indkktviolation);
    
    if nbkktviolation==0
        break;
    end;
    
    % créer le nouveau working set
    workingset=zeros(taille,1);
    if nbkktviolation >=qpsize
        randomindice=randperm(nbkktviolation);
        workingset(indkktviolation(randomindice(1:min(nbkktviolation,qpsize))))=1;
    else
        indkktgood=find(~kktviolation);
        workingset(indkktviolation)=1;
        randomindicegood=randperm(length(indkktgood));
        %   keyboard
        workingset(indkktgood(randomindicegood(1:min(length(indkktgood),qpsize-nbkktviolation))))=1;
    end;
    
    indworkingset=find(workingset);
    nws=~workingset;
    indnws= find(nws);
    
    %----- Pour le calcul de QbAlphaN on cherche que les alphaN non nulles
    nwSV= (nws & SV);
    indnwSV=find(nwSV);
    Qbnalphan=0;
    if length(indnwSV)>0
        
        chunks=ceil(length(indnwSV)/chunksize);
        for ch=1:chunks
            ind=(1+(ch-1)*chunksize ): min( length(indnwSV), ch*chunksize);
            
            
            %   Kchunk=svmkernel(x(indpos(matriceind(:,1)),:),kernel,kerneloption)-svmkernel(x(indpos(matriceind(:,1)),:),kernel,kerneloption,x(indneg(matriceind(:,2)),:))...
            %       -svmkernel(x(indneg(matriceind(:,2)),:),kernel,kerneloption,x(indpos(matriceind(:,1)),:))+svmkernel(x(indneg(matriceind(:,2)),:),kernel,kerneloption);
            
            Kchunk=svmkernel(x(indpos(matriceind(indworkingset,1)),:),kernel,kerneloption,x(indpos(matriceind(indnwSV(ind),1)),:))...
                -svmkernel(x(indpos(matriceind(indworkingset,1)),:),kernel,kerneloption,x(indneg(matriceind(indnwSV(ind),2)),:))...      
                -svmkernel(x(indneg(matriceind(indworkingset,2)),:),kernel,kerneloption,x(indpos(matriceind(indnwSV(ind),1)),:))...
                +svmkernel(x(indneg(matriceind(indworkingset,2)),:),kernel,kerneloption,x(indneg(matriceind(indnwSV(ind),2)),:));
            
            
            
            Qbnalphan=Qbnalphan +Kchunk*alpha(indnwSV(ind));
        end;
        e= - (Qbnalphan - margin*ones(qpsize,1));
        
    else
        e=ones(qpsize,1)*margin;
    end;
    Kbb=svmkernel(x(indpos(matriceind(indworkingset,1)),:),kernel,kerneloption)-svmkernel(x(indpos(matriceind(indworkingset,1)),:),kernel,kerneloption,x(indneg(matriceind(indworkingset,2)),:))...
        -svmkernel(x(indneg(matriceind(indworkingset,2)),:),kernel,kerneloption,x(indpos(matriceind(indworkingset,1)),:))+svmkernel(x(indneg(matriceind(indworkingset,2)),:),kernel,kerneloption);
    
    verboseqp=0;
    [alphab, lambdab, pos]=monqp(Kbb,e,zeros(qpsize,1),0,C,lambda,verboseqp);
    
    alphaold=alpha;
    aux=zeros(qpsize,1);
    aux(pos)=alphab;
    alpha(indworkingset)=aux;
    iteration=iteration+1;
    
    if verbose==1
        fprintf('i: %d number changedAlpha : %d  Nb KKT Violation: %d\n',iteration,length(find( abs(alpha-alphaold)> difftol)),nbkktviolation);
    elseif verbose==2
        fprintf('.'); 
    end
end;
%keyboard
pos=find(alpha>0);
w=alpha(pos);
% pos=[find(abs(kkt)<kkttol)';find(kkt<-kkttol)'];
% w=alpha(pos);

w0=0;
span=1;
xsuppos=x(indpos(matriceind(pos,1)),:);
xsupneg=x(indneg(matriceind(pos,2)),:);
xsup={xsuppos xsupneg};
if verbose==1
fprintf('\noptimization done\n');
end
