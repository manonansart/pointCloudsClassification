function [xsup,w,b,pos,timeps,alpha]=svmclassL2LS(x,y,c,lambda,kernel,kerneloption,verbose,span,qpsize,chunksize)


% 
% [xsup,w,b,pos,timeps,alpha]=svmclassL2LS(x,y,c,lambda,kernel,kerneloption,verbose,span,qpsize,chunksize)
%
% %
% %   large-scale classification svm 
% %


% dbstop if warning
% dbstop if error

n=size(y,1);
if nargin < 10
    chunksize=100;
end;
if nargin<9
    qpsize=100; 
end;
if nargin < 10
    % even number
    chunksize=qpsize;
    
end;
if isstruct(x)
    if length(x.indice)~=length(y)
        error('Length of x and y should be equal');
    end;
end


maxqpsize=qpsize;
if qpsize> n
    qpsize=n;
end;
if rem(qpsize,2)==1
    qpsize=qpsize-1;
end;
kkttol=1e-5;
difftol=1e-12;





alphaold=zeros(n,1);
alpha=zeros(n,1);
workingset=zeros(n,1);
nws=zeros(n,1);

class1=(y>=0);
class0=(y<0);
iteration=0;
bias=0;
lambdab=0;
%
%keyboard

while 1
    
        if iteration==200
         % keyboard
            %break    
        end;
    
    
    %
    %   calcul des indices des SV et non SV
    %
    
    SV=(abs(alpha)>difftol);
    
    
    %
    %    Calcul de la sortie du SVM
    %
    
    if iteration==0  ;
        changedSV=find(SV);
        changedAlpha=alpha(changedSV);
        s=zeros(n,1);
        
    else
        changedSV=find( abs(alpha-alphaold)> difftol );
        changedAlpha=alpha(changedSV)-alphaold(changedSV);  
    end;
    
    if ~isempty(changedSV)
        
        chunks1=ceil(n/chunksize);
        chunks2=ceil(length(changedSV)/chunksize);
        
        for ch1=1:chunks1
            ind1=(1+(ch1-1)*chunksize) : min( n, ch1*chunksize);
            for ch2=1:chunks2
                ind2=(1+(ch2-1)*chunksize) : min(length(changedSV), ch2*chunksize);
                
                %-----------------------------------------------------------                
                if ~isfield(x,'datafile')
                    x1=x(ind1,:);
                    x2=x(changedSV(ind2),:);
                else
                    x1=fileaccess(x.datafile,x.indice(ind1),x.dimension);
                    x2=fileaccess(x.datafile,x.indice(changedSV(ind2)),x.dimension);
                    
                end;   
                kchunk=svmkernel(x1,kernel,kerneloption,x2);
                %-----------------------------------------------------------  
                %kchunk=svmkernel(x(ind1,:),kernel,kerneloption,x(changedSV(ind2),:)); 
                %-----------------------------------------------------------  
                coeff=changedAlpha(ind2).*y(changedSV(ind2));
                
                s(ind1)=s(ind1)+ kchunk*coeff;
            end;
        end
        
    end;    
    
    %
    %  calcul du biais du SVM que sur l'ensemble du working set et
    %  SVnonbound
    
%         indworkingSV= find(SV& workingset);
%         if ~isempty(indworkingSV)
%           %  bias= mean( y(indworkingSV)-s(indworkingSV) );   
%             bias= mean( y(indworkingSV)-s(indworkingSV) - y(indworkingSV).*alpha(indworkingSV)/c );
%         end;
        
        
        if sum(SV)>0
          %  bias= mean( y(indworkingSV)-s(indworkingSV) );   
            bias= mean( y(SV)-s(SV) - y(SV).*alpha(SV)/c );
        else
            bias=0;
        end;

   % bias=lambdab; % this is the Lagrange multiplier of the equality constraints of monqp
    
    
    %
    %  KKT Conditions
    %
    
    kkt=(s+bias).*y - 1;
      testkkt= abs(abs(kkt)-alpha/c);  %  check the equation of KKT for this condition.
    kktviolation=   (SV   & ( testkkt> kkttol) )|( ~SV & (kkt < -kkttol));
%  kktviolation=   (SV   & ( testkkt> kkttol) )|( ~SV & (kkt +alpha/c< -kkttol));
%     testkkt=-kkt -alpha/c;
%     kktviolation1= (SV   & ( testkkt> kkttol) );
%     kktviolation2=( ~SV & (kkt < - kkttol));
%    kktviolation=   kktviolation1 | kktviolation2;
%     
    if sum(kktviolation)==0
        break;   %  c'est fini tout 
    end;
    
    
    
    %
    %   Calcul du nouveau working set
    %
    
    if iteration==0
        searchdir=rand(n,1);
        set1=class1;
        set2=class0;
    end;
    
    
    
    oldworkingset=workingset;
    workingset=zeros(n,1);
    n1=sum(set1);
    n2=sum(set2);
    
    
    %         indpos=find(y==1);
    %         indneg=find(y==-1);
    %         
    %         
    %         % ici on fait un tirage aléatoire parmi tout les points!! c vraiment
    %         % tout pourri.
    %         RandIndpos=randperm(length(indpos));
    %         RandIndneg=randperm(length(indneg));
    %         nbpos=min(length(indpos),round(qpsize/2));
    %         nbneg=min(length(indneg),round(qpsize/2));
    %         ind=[indpos(RandIndpos(1:nbpos));indneg(RandIndneg(1:nbneg))];
    %         workingset(ind)=ones(length(ind),1);
    %         
    
    indkktviolation=find(kktviolation);
    nbkktviolation=sum(kktviolation);
    if qpsize==n
        workingset=ones(n,1);
%     elseif nbkktviolation <=qpsize
%         nbOK=qpsize-nbkktviolation;
%         indOK=find(~kktviolation);
%         indiceOK=randperm(length(indOK));
%         
%         ind=[indkktviolation; indOK(indiceOK(1:nbOK))];
%         workingset(ind)=ones(length(ind),1);
    else
        
        indposkktviol= find(y==1 & kktviolation);
        indposkktviol=indposkktviol(randperm(length(indposkktviol)));
        indnegkktviol= find(y==-1 & kktviolation);
        indnegkktviol=indnegkktviol(randperm(length(indnegkktviol)));
        indposOK= find(y==1 &  ~kktviolation);
        indnegOK= find(y==-1 &  ~kktviolation);
        nbposViol=min(length(indposkktviol),round(qpsize/2));
        nbnegViol=min(length(indnegkktviol),round(qpsize/2));
        nbposOK=min(qpsize/2-nbposViol,length(indposOK));;
        nbnegOK=min(qpsize/2-nbnegViol,length(indposOK));
        ind=[indposkktviol(1:nbposViol);indposOK(1:nbposOK) ;indnegkktviol(1:nbnegViol);indnegOK(1:nbnegOK)];
        workingset(ind)=ones(length(ind),1);
        
        %     indkktviolation=find(kktviolation);
        %     nbkktviolation=length(indkktviolation);
        %             randomindice=randperm(nbkktviolation);
        %         workingset(indkktviolation(randomindice(1:min(nbkktviolation,qpsize))))=1;
        
        
    end;
    if all( abs(oldworkingset-workingset) < difftol)
        indpos=find(y==1);
        indneg=find(y==-1);
        %keyboard
        
        % ici on fait un tirage aléatoire parmi tout les points!! c vraiment
        % tout pourri.
        RandIndpos=randperm(length(indpos));
        RandIndneg=randperm(length(indneg));
        nbpos=min(length(indpos),round(qpsize/2));
        nbneg=min(length(indneg),round(qpsize/2));
        ind=[indpos(RandIndpos(1:nbpos));indneg(RandIndneg(1:nbneg))];
        workingset(ind)=ones(length(ind),1);
        
    end;  
    
    
    indworkingset=find(workingset);
    workingsize=length(indworkingset);
    nws=~workingset;
    indnws= find(nws);
    
    
    
    
    %
    %   Resolution du QP probleme sur le nouveau Working set
    %
    
    % le calcul de Qbn*alphan ne fait intervenir que les donnÃ©es aux alphan non nulles et les donnÃ©es de la working
    % set
    
    
    nwSV= (nws & SV);
    indnwSV=find(nwSV);
    Qbnalphan=0;
    if length(indnwSV)>0
        
        chunks=ceil(length(indnwSV)/chunksize);
        for ch=1:chunks
            ind=(1+(ch-1)*chunksize ): min( length(indnwSV), ch*chunksize);
            %-----------------------------------------------------------                
            if ~isfield(x,'datafile')
                x1=x(indworkingset,:);
                x2=x(indnwSV(ind),:);
            else
                x1=fileaccess(x.datafile,x.indice(indworkingset),x.dimension);
                x2=fileaccess(x.datafile,x.indice(indnwSV(ind)),x.dimension);
                
            end;   
            pschunk=svmkernel(x1,kernel,kerneloption,x2);
            %-----------------------------------------------------------  
            % pschunk=svmkernel(x(indworkingset,:),kernel,kerneloption,x(indnwSV(ind),:));
            %-----------------------------------------------------------  
            
            
            
            Qbnalphan=Qbnalphan + y(indworkingset).*(pschunk*(alpha(indnwSV(ind)).*y(indnwSV(ind))));
        end;
        e= - (Qbnalphan - ones(workingsize,1));
        
    else
        e=ones(workingsize,1);
    end;
    
    %-----------------------------------------------------------  
    % Calcul de la matrice Hbb
    %-----------------------------------------------------------    
    yb=y(indworkingset);
    if ~isfield(x,'datafile')
        psbb=svmkernel(x(indworkingset,:),kernel,kerneloption);
    else
        x1=fileaccess(x.datafile,x.indice(indworkingset),x.dimension);
        psbb=svmkernel(x1,kernel,kerneloption);
    end;
    Hbb=psbb.*(yb*yb')+1/c*eye(size(psbb));
    
    
    A=yb;
    if length(indnws)>0
        b=-alpha(indnws)'*y(indnws);
    else
        b=0;
    end;
    
    cinfty=+inf;
    [alphab,lambdab,pos]=monqp(Hbb,e,A,b,cinfty,lambda,0);%,psbb);
    %     [alphab,lambdab,pos]=monqpCinfty(Hbb,e,A,b,lambda,0);%,psbb);
    alphaold=alpha;
    aux=zeros(workingsize,1);
    aux(pos)=alphab;
    alpha(indworkingset)=aux;
    iteration=iteration+1;
    if verbose >0
        obj= 0.5*aux'*Hbb*aux- aux'*e;
        fprintf('i: %d number changedAlpha : %d  Nb KKT Violation: %d Objective Val:%f\n',iteration,length(find( abs(alpha-alphaold)> difftol)),sum(kktviolation),obj);
    end;
    if sum(kktviolation) < maxqpsize
        qpsize=maxqpsize;
        chunksize=maxqpsize;
    end;
end;


SV=(abs(alpha)>difftol);


pos=find(SV);

if ~isfield(x,'datafile')
    xsup = x(pos,:);
else
    xsup=x;
    xsup.indice=x.indice(pos);
end;
ysup = y(pos);
w = (alpha(pos).*ysup);

indworkingSV= find(SV& workingset);
% if ~isempty(indworkingSV)
%    % bias= mean( y(indworkingSV)-s(indworkingSV) );   
%     bias= mean( y(indworkingSV)-s(indworkingSV) - y(indworkingSV).*alpha(indworkingSV)/c );
% end;

        if ~isempty(SV)
          %  bias= mean( y(indworkingSV)-s(indworkingSV) );   
            bias= mean( y(SV)-s(SV) - y(SV).*alpha(SV)/c );
        else
            bias=0;
        end;

b = bias;
timeps=[];
alpha=alpha(pos);
