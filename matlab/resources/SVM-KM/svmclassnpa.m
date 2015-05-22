function  [xsup,alpha,b,pos]=svmclassnpa(x,y,C,kernel,kerneloption,verbose);

% USAGE 
%   [xsup,alpha,b,pos]=svmclassnpa(x,y,C,kernel,kerneloption,verbose); 
%
%
%    Main ROUTINE For Nearest Point Algorithm
%
%   This function solve the SVM Classifier problem by 
%   transforming the soft-margin problem into a hard margin problem
%   by means of a slight modification of the kernel and the
%   introduction of a quadratic penalization term.
%
%   The problem is equivalent  to look for the nearest points of 
%   two convex hulls. This is solved by the point of minimum norm is the 
%   Minkowski set (U-V) where U is the class 1 set and V the class 2 set 
%   
%   This method is deeply described is 
%   
%   Tech Rep : A Fast Iterative NPA for SVM Classifier
%   S. Keerthi et al
% 

%   Last Modification : 31/03/2001  A. R
%


global  n;
global  m;
% global  C;
% global  kernel;
% global  kerneloption;
% global  x;
% global  y;
global  deluu;
global  delvv;
global  deluv;
global  delzz;
global  eu;
global  ev;
global  alind;
global  beta;




%  Initialization


eps1=0.001; %type I loop
eps=0.001; %type II loop

m=length(y);
beta=zeros(1,m);
eu=zeros(7,m);
ev=zeros(1,m);

indpos=find(y==1);
indmoins=find(y==-1);
i=indpos(1);
j=indmoins(1);
beta(i)=1;
beta(j)=1;
alind(i)=1;
alind(j)=1;
nsupp=2;
deluu=svmkernel(x(i,:),kernel,kerneloption,x(i,:))+1/C;
delvv=svmkernel(x(j,:),kernel,kerneloption,x(j,:))+1/C;
deluv=svmkernel(x(i,:),kernel,kerneloption,x(j,:));
delzz=deluu+delvv-2*deluv;
eu(i)=deluu;
eu(j)=deluv;
ev(i)=deluv;
ev(j)=delvv;
Examine_NonSV=1;SV_Optimality=1;NonSV_Optimality=0;toohigheps=1;
Num_Type4_Updates=0;Num_Type2_Updates=0;

%-------------------------------------------------------------------%
%                                                                   %
%                               MAIN LOOP                           %
%                                                                   %
%-------------------------------------------------------------------%

while (SV_Optimality==0 | NonSV_Optimality==0 ) 
    ind=find(beta<=0);
    alind(ind)=zeros(size(ind));
    beta(ind)=zeros(size(ind));
    if Examine_NonSV==1             % TYPE 1 LOOOP
        Num_Type1_Updates=0;
        NonSV_Optimality=1;
        Examine_NonSV=0;
        ind_NonSV=find(alind==0);
        
        for k=(ind_NonSV)
            ind_I=intersect(find(y==1),find(alind==1));       % Positive class
            ind_J=intersect(find(y==-1),find(alind==1));      % Negative class  
           % keyboard
            eu(k)=sum(beta(ind_I).*(svmkernel(x(k,:),kernel,kerneloption,x(ind_I,:))+(ind_I==k)./C));
            ev(k)=sum(beta(ind_J).*(svmkernel(x(k,:),kernel,kerneloption,x(ind_J,:))+(ind_J==k)./C));
            zdk=eu(k)-ev(k);
            zdu=deluu-deluv;
            zdv=deluv-delvv;  
            if ( (y(k)==1 & (zdu-zdk)>=0.5*eps1*delzz ) | (y(k)==-1 & (zdk-zdv)>=0.5*eps1*delzz) )
                NonSV_Optimality=0;
                beta(k)=0;
                alind(k)=1;
                success=Take_Step(k,x,y,C,kernel,kerneloption);
                if delzz<0         
                    keyboard
                end;
                if success==0
                    alind(k)=0;
                else
                    Num_Type1_Updates=Num_Type1_Updates+1;
                end;
            end;
            
        end;
        
        
    else
        Examine_NonSV=1;            % Support Vector Processing....  
        nsupp_old=nsupp;
        nsupp=sum(alind==1);
        if abs(nsupp_old-nsupp) >=0.05*nsupp
            Max_Updates=m;
        else
            Max_Updates=10*m;
        end;
        Loop_Completed=0;
        Num_Type2_Updates=0;
        nbloop=0;
        while(Loop_Completed==0)
            nbloop=nbloop+1;
            zdu=deluu-deluv;
            zdv=deluv-delvv;
            ind_I=intersect(find(y==1),find(alind==1));       % Positive SV
            ind_J=intersect(find(y==-1),find(alind==1));      % Negative SV
            zdi=eu(ind_I)-ev(ind_I);
            [maxi imax]=max(zdu-zdi);
            imax=ind_I(imax);
            zdj=eu(ind_J)-ev(ind_J);
            [maxi jmax]=max(zdj-zdv);
            jmax=ind_J(jmax);
            gu=zdu-eu(imax)+ev(imax);
            gv=eu(jmax)-ev(jmax)-zdv;
            if (gu<=0.5*eps*delzz) & (gv <=0.8*eps*delzz)
                Loop_Completed=1;
                SV_Optimality=1; 
            else
                if (gu >=gv) 
                    kmax=imax;
                else
                    kmax=jmax;
                end;
                %[success,cas]=Take_Step(kmax);
                success=Take_Step(kmax,x,y,C,kernel,kerneloption);
                if delzz<0          
                    keyboard
                end;
                if success==0
                    Loop_Completed=1;
                    SV_Optimality=0;
                else
                    
                    Num_Type2_Updates=Num_Type2_Updates+1;
                end;
                if Num_Type2_Updates>Max_Updates
                    Loop_Completed=1;
                    SV_Optimality=0;
                end;
            end;
            
        end;
    end;
    
    if SV_Optimality & NonSV_Optimality  
        fprintf('Optimality criteria Satisfied...\n');
 
        pos=find(alind==1);
        xsup=x(pos,:);
        ysup=y(pos,:);
        if ~isempty(imax) & ~isempty(imax)
            h_U = ev(imax) - eu(imax);
            h_V = eu(jmax) - ev(jmax);
            gamma = 2.0 / (-h_U - h_V);
            b  = (h_V - h_U) / (h_V + h_U);
            alpha=beta(pos).*gamma.*y(pos)';
            alpha=alpha(:);
        end;
        return
    end;
    
    if Num_Type1_Updates==0 & Num_Type2_Updates==0 
        fprintf('Algorithm has not converged...\n');
        pos=find(alind==1);
        xsup=x(pos,:);
        ysup=y(pos,:);
        if ~isempty(imax) & ~isempty(imax)
            h_U = ev(imax) - eu(imax);
            h_V = eu(jmax) - ev(jmax);
            gamma = 2.0 / (-h_U - h_V);
            b  =(h_V - h_U) / (h_V + h_U);
            alpha=beta(pos).*gamma.*y(pos)';
            alpha=alpha(:);
        end;
        return
    end;
    
    
    if ~isempty(verbose) & verbose==1
    fprintf(' ||z||= %f SV_Optimality:%d  NonSV_Optimality:%d\n',delzz,SV_Optimality, NonSV_Optimality); 
end;
    
    
end;



pos=find(alind==1);
xsup=x(pos,:);
ysup=y(pos,:);
h_U = ev(imax) - eu(imax);
h_T = eu(jmax) - ev(jmax);
gamma = 2.0 / (-h_U - h_V);
b  =(h_V - h_U) / (h_N + h_U);
alpha=(beta(pos).*gamma.*y(pos)');
alpha=alpha(:);



%---------------------------------------------------------------------%
%                                                                     %  
%                           Function Take_Step                        %  
%                                                                     %  
%---------------------------------------------------------------------%
function [success,cas]=Take_Step(kmax,x,y,C,kernel,kerneloption)

global   n;
global  m;
% global C;
% global kernel;
% global kerneloption;
% global  x;
% global y;
global   deluu;
global   delvv;
global   deluv;
global   delzz;
global   eu;
global   ev;
global   alind;
global   beta;


% compute kmin%
success=0;
zdu=deluu-deluv;
zdv=deluv-delvv;
ind_betapos=find(beta>=0);
ind_I=intersect(find(y==1),find(alind==1));      % Positive SV
ind_J=intersect(find(y==-1),find(alind==1));     % Negative SV   
ind_Ibetapos=intersect(ind_I,ind_betapos);
ind_Jbetapos=intersect(ind_J,ind_betapos);
[minii,imin]=min(zdu-eu(ind_Ibetapos)+ev(ind_Ibetapos));
imin=ind_Ibetapos(imin);
[minij,jmin]=min(-zdv+eu(ind_Jbetapos)-ev(ind_Jbetapos));
jmin=ind_Jbetapos(jmin);
if (minii)<(minij)
    kmin=imin;
else
    kmin=jmin;
end;








% Do Modified Gilbert Step if beta(kmin)=1

if (beta(kmin)>=1 ) | kmin==0        
    cas=5;
    success=Gilbert_Step(kmax,x,y,C,kernel,kerneloption);
    return;
end;
mu=beta(kmin)/(1-beta(kmin));
%kmin
%kmax
if isempty(kmin)
    kmin=kmax;
end;
%kmin
%kmax
%size(x)
psi = svmkernel(x(kmin,:),kernel,kerneloption,x(kmax,:))+(kmin==kmax)./C;
psi1= svmkernel(x(kmax,:),kernel,kerneloption,x(kmax,:))+(kmax==kmax)./C;
psi2= svmkernel(x(kmin,:),kernel,kerneloption,x(kmin,:))+(kmin==kmin)./C;
eumax = eu(kmax);
evmax = ev(kmax);
eumin = eu(kmin);
evmin = ev(kmin);

if y(kmax)==1 & y(kmin)==1
    cas=1;
    imax=kmax; imin=kmin;
    d11=delzz;
    t1=deluu - eumin - deluv + evmin;
    d22 = delzz + mu*mu*(deluu + psi2 - 2.0*eumin) + 2.0*mu*t1;
    d33 = psi1 + delvv - 2.0*evmax;
    d12 = delzz + mu*t1;
    d13 = eumax - deluv - evmax + delvv;
    d23 = d13 + mu*(eumax - deluv - psi + evmin);
    [d,la1,la2,la3,flag]=triangle(d11,d22,d33,d12,d13,d23);
    if d >=delzz | d < 0
        success=0;
        return;
    end;
    success=1;
    delzz=d;
    la1b = la1 + la2 + la2*mu;
    la2b = -la2*mu;
    la3b = la3;
    deluu = la1b*la1b*deluu + la2b*la2b*psi2 + la3b*la3b*psi1 + 2.0*la1b*la2b*eumin ;
    deluu = deluu + 2.0*la1b*la3b*eumax +2.0*la2b*la3b*psi;
    deluv = la1b*deluv + la2b*evmin + la3b*evmax;
    ind_SV=find(alind==1);
    kkmin=svmkernel(x(imin,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==imin)./C;
    kkmax=svmkernel(x(imax,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==imax)./C;
    eu(ind_SV)=la1b*eu(ind_SV)+la2b*kkmin+la3b*kkmax;
    ind_I=intersect(find(y==1),find(alind==1));
    beta(ind_I)=la1b*beta(ind_I);
    beta(imax)=beta(imax)+la3b;
    if (flag ==2 | flag== 5 | flag == 3) 
        alind(imin)=0;
        beta(imin)=0;
    else
        beta(imin) = beta(imin) + la2b;
    end;
end;
if y(kmax)==-1 & y(kmin)==-1
    cas=2; 
    jmax=kmax;jmin=kmin;
    d11 = delzz;
    t1 = delvv - evmin - deluv + eumin; 
    d22 = delzz + mu*mu*(delvv + psi2 - 2.0*evmin) + 2.0*mu*t1;
    d33 = psi1 + deluu - 2.0*eumax;
    d12 = delzz + mu*t1;
    d13 = evmax - deluv - eumax + deluu;
    d23 = d13 + mu*(eumin - deluv - psi + evmax);
    [d,la1,la2,la3,flag]=triangle(d11,d22,d33,d12,d13,d23);
    if d >=delzz | d < 0
        success=0;
        return;
    end;
    success = 1;
    delzz = d;
    la1b = la1 + la2 + la2*mu;
    la2b = -la2*mu;
    la3b = la3;
    delvv = la1b*la1b*delvv + la2b*la2b*psi2 + la3b*la3b*psi1 +   2.0*la1b*la2b*evmin;
    delvv=delvv+ 2.0*la1b*la3b*evmax +  2.0*la2b*la3b*psi;
    deluv = la1b*deluv + la2b*eumin + la3b*eumax;
    ind_SV=find(alind==1);
    kkmin=svmkernel(x(jmin,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==jmin)./C;
    kkmax=svmkernel(x(jmax,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==jmax)./C;
    ev(ind_SV)=la1b*ev(ind_SV)+la2b*kkmin+la3b*kkmax;
    ind_J=intersect(find(y==-1),find(alind==1));
    beta(ind_J)=la1b*beta(ind_J);
    beta(jmax)=beta(jmax)+la3b;
    if (flag ==2 | flag== 5 | flag == 3) 
        alind(jmin)=0;
        beta(jmin)=0;
    else
        beta(jmin) = beta(jmin) + la2b;
    end;
end;
if y(kmax)==1 & y(kmin)==-1
    cas=3; 
    imax=kmax;jmin=kmin;
    r = mu*(evmax - psi - deluv + eumin);
    s1 = (evmax - eumax - deluv + deluu);
    s2 = mu*(delvv - deluv - evmin + eumin);
    d1 = psi1 + deluu - 2.0*eumax;
    d2 = mu*mu*(delvv + psi2 - 2.0*evmin);
    dold = delzz;
    [d,la1,la2,flag,vert1,vert2]=twolines(r,s1,s2,d1,d2,dold);
    if flag==0 | d>=delzz | d < 0
        
        
        success=0;
        return;
    end;
    success=1;
    delzz=d;
    deluu = (1.0 - la1)*(1.0 -la1)*deluu + la1*la1*psi1 + 2.0*(1.0- la1)*la1*eumax;
    delvv = (1.0 + la2*mu)*(1.0 + la2*mu)*delvv +la2*la2*mu*mu*psi2 -  2.0*(1.0 + la2*mu)*la2*mu*evmin;
    deluv = (1.0- la1)*(1.0 + la2*mu)*deluv - (1.0 - la1)*la2*mu*eumin +  (1.0 + la2*mu)*la1*evmax - la1*la2*mu*psi;
    ind_SV=find(alind==1);
    eu(ind_SV)=(1-la1)*eu(ind_SV)+la1*(svmkernel(x(imax,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==imax)./C);
    ev(ind_SV)=(1+la2*mu)*ev(ind_SV)-la2*mu*(svmkernel(x(jmin,:),kernel,kerneloption,x(ind_SV,:))++(ind_SV==jmin)./C);
    ind_I=intersect(find(y==1),find(alind==1));
    beta(ind_I)=(1-la1)*beta(ind_I);
    ind_J=intersect(find(y==-1),find(alind==1));
    beta(ind_J)=(1+la2*mu)*beta(ind_J);
    beta(imax)=beta(imax)+la1;
    if (vert2==2);
        alind(jmin)=0;
        beta(jmin)=0;
    else
        beta(jmin)=beta(jmin)-la2*mu;
    end;
end;
if y(kmax)==-1 & y(kmin)==1
    cas=4; 
    jmax = kmax;
    imin = kmin;
    r = mu*(eumax - psi - deluv + evmin);       %OK
    s1 = (eumax - evmax - deluv + delvv);       %OK
    s2 = mu*(deluu - deluv - eumin + evmin);    %OK
    d1 = psi1 + delvv - 2.0*evmax;              %OK
    d2 = mu*mu*(deluu + psi2 - 2.0*eumin);      %OK
    dold = delzz;
    
    [d,la1,la2,flag,vert1,vert2]=twolines(r,s1,s2,d1,d2,dold);
    if flag==0 | d>=delzz | d < 0
        
        success=0;
        return;
    end;
    success=1;
    delzz=d;
    delvv = (1.0 - la1)*(1.0 -la1)*delvv + la1*la1*psi1 + 2*(1- la1)*la1*evmax;
    deluu = (1.0 + la2*mu)*(1 + la2*mu)*deluu +  la2*la2*mu*mu*psi2 -  2.*(1 + la2*mu)*la2*mu*eumin;
    deluv = (1- la1)*(1+ la2*mu)*deluv -(1 - la1)*la2*mu*evmin + (1 + la2*mu)*la1*eumax - la1*la2*mu*psi;
    ind_SV=find(alind==1);
    ev(ind_SV)=(1-la1)*ev(ind_SV)+la1*(svmkernel(x(jmax,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==jmax)./C);
    eu(ind_SV)=(1+la2*mu)*eu(ind_SV)-la2*mu*(svmkernel(x(imin,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==imin)./C);
    ind_J=intersect(find(y==-1),find(alind==1));
    beta(ind_J)=(1-la1)*beta(ind_J);
    ind_I=intersect(find(y==1),find(alind==1));
    beta(ind_I)=(1+la2*mu)*beta(ind_I);
    beta(jmax)=beta(jmax)+la1;
    if (vert2==2);
        alind(imin)=0;
        beta(imin)=0;
    else
        beta(imin)=beta(imin)-la2*mu;
    end;
end;
return;

%---------------------------------------------------------------------%
%                                                                     %  
%                           Function Gilbert_Step                     %  
%                                                                     %  
%---------------------------------------------------------------------%
function success=Gilbert_Step(kmax,x,y,C,kernel,kerneloption);




global  n;
% global  C;
% global  kernel;
% global  kerneloption;
% global  x;
% global  y;
global  deluu;
global  delvv;
global  deluv;
global  delzz;
global  eu;
global  ev;
global  alind;
global  beta;



if y(kmax)==1 
    imax = kmax;
    d11 = delzz;
    dimax=svmkernel(x(imax,:),kernel,kerneloption,x(imax,:))+(imax==imax)./C;
    d22 = dimax + delvv - 2.0*ev(imax);
    d12 = eu(imax) - ev(imax) - deluv + delvv;
    [d,lambda]=linesegment(d11,d22,d12);
    if (d >= delzz) | d < 0
        
        success=0;
        return;
    end;
    success=1;
    delzz=d;
    om = 1.0 - lambda;
    deluu = om*om*deluu + lambda*lambda*dimax +  2*lambda*om*eu(imax);
    deluv = om*deluv + lambda*ev(imax);
    ind_SV=find(alind==1);
    eu(ind_SV)=om*eu(ind_SV)+lambda*(svmkernel(x(imax,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==imax)./C);
    
    ind_I=intersect(find(y==1),find(alind==1));
    beta(ind_I)=om*beta(ind_I);
    beta(imax)=beta(imax)+lambda;
    
    
    
else
    jmax=kmax;
    d11 = delzz;
    djmax=svmkernel(x(jmax,:),kernel,kerneloption,x(jmax,:))+(jmax==jmax)./C;
    d22 = djmax + deluu - 2.0*eu(jmax);
    d12 = ev(jmax) - eu(jmax) - deluv + deluu;
    [d,lambda]=linesegment(d11,d22,d12);
    if (d >= delzz) | d < 0
        
        
        success=0;
        return;
    end;
    success=1;
    delzz=d;
    om = 1.0 - lambda;
    delvv = om*om*delvv + lambda*lambda*djmax + 2*lambda*om*ev(jmax);
    deluv = om*deluv + lambda*eu(jmax);
    ind_SV=find(alind==1);
    ev(ind_SV)=om*ev(ind_SV)+lambda*(svmkernel(x(jmax,:),kernel,kerneloption,x(ind_SV,:))+(ind_SV==jmax)./C);
    ind_J=intersect(find(y==-1),find(alind==1));
    beta(ind_J)=om*beta(ind_J);
    beta(jmax)=beta(jmax)+lambda;
end;
return

%---------------------------------------------------------------------%
%                                                                     %  
%                           Function Triangle                         %  
%                                                                     %  
%---------------------------------------------------------------------%



function [d,lambda1,lambda2,lambda3,flag]=triangle(d11,d22,d33,d12,d13,d23)

%   Tech Rep : A Fast Iterative NPA for SVM Classifier
%   S. Keerthi et al
% 
%   23/03/2001


[d,lambda2,vert]=linesegment(d11,d22,d12);
lambda1=1-lambda2;
lambda3=0;
if (vert==1) 
    flag = 1;
else 
    if (vert== 2) 
        flag = 2;
    else
        flag = 4;
    end;
end;



[dtilde,lambda3tilde,vert]=linesegment(d22,d33,d23);
if dtilde<d 
    d=dtilde;
    lambda1=0;
    lambda2=1-lambda3tilde;
    lambda3=lambda3tilde;
    if (vert==1) 
        flag = 2;
    else 
        if (vert== 2) 
            flag = 3;
        else
            flag = 5;
        end;
    end;
    
end;

[dtilde,lambda1tilde,vert]=linesegment(d33,d11,d13);
if dtilde<d 
    d=dtilde;
    lambda1=lambda1tilde;
    lambda2=0;
    lambda3=1-lambda1tilde;   
    if (vert==1) 
        flag = 3;
    else 
        if (vert== 2) 
            flag = 1;
        else
            flag = 6;
        end;
    end;
    
end;

e11=d22+d11-2*d12;
e22=d33+d11-2*d13;
e12=d23-d12-d13+d11;
den=e11*e22-e12^2;
if den <=0
    return
end;
f1=d11-d12;
f2=d11-d13;
lambda2tilde=(e22*f1-e12*f2)/den;
lambda3tilde=(-e12*f1+e11*f2)/den;
lambda1tilde=1-lambda2tilde-lambda3tilde;
dtilde=d11-lambda2tilde*f1-lambda3tilde*f2;
if (lambda1tilde>0) & (lambda2tilde>0) & (lambda3tilde>0) & (dtilde<d)
    d=dtilde;
    lambda1=lambda1tilde;
    lambda2=lambda2tilde;
    lambda3=lambda3tilde;
    flag=0;
end;

%---------------------------------------------------------------------%
%                                                                     %  
%                           Function Twolines                         %  
%                                                                     %  
%---------------------------------------------------------------------%

function [d,lambda1,lambda2,flag,vert1,vert2]=twolines(r,s1,s2,d1,d2,dold);

%   Tech Rep : A Fast Iterative NPA for SVM Classifier
%   S. Keerthi et al
% 
%   23/03/2001



if (d1<=0) | (d2<=0)
    d=[];
    lambda1=[];
    lambda2=[];
    vert1=[];
    vert2=[];
    flag=0;
    
    
    return;
end;
den=d1*d2-r.^2;
if den <=0
    d=[];
    lambda1=[];
    lambda2=[];
    vert1=[];
    vert2=[];
    flag=0;
    
    return;
end;
lambda1=(s1*d2-s2*r)/den;
if lambda1<0
    lambda1=0;
    vert1=0;
else
    if lambda1>1
        lambda=1;
        vert1=2;
    else
        vert1=0;
    end;
end,
lambda2=(lambda1*r-s2)/d2;
if lambda2<0
    lambda2=0;
    vert2=1;
else
    if lambda2>1
        lambda2=1;
        vert2=2;
    else
        d=dold+d1*lambda1^2+d2*lambda2^2-2*r*lambda1*lambda2+2*s2*lambda2-2*s1*lambda1;
        flag=1;
        vert2=0;
        return;
    end;
end,
lambda1=(lambda2*r+s1)/d1;
if lambda1<0
    lambda1=0;
    vert1=1;
else
    if lambda1>1
        lambda1=1;
        vert1=2;
    else
        vert1=0;
    end;
end;
d=dold+d1*lambda1^2+d2*lambda2^2-2*r*lambda1*lambda2+2*s2*lambda2-2*s1*lambda1;
flag=1;


%---------------------------------------------------------------------%
%                                                                     %  
%                           Function LineSegment                      %  
%                                                                     %  
%---------------------------------------------------------------------%


function  [d,lambda,vert]=linesegment(d11,d22,d12);


%   
%   Tech Rep : A Fast Iterative NPA for SVM Classifier
%   S. Keerthi et al
% 
%   23/03/2001

d=d11;
lambda=0;
vert=1;
if d22< d
    d=d22;
    lambda=1;
    vert=2;
end;
den=d11+d22-2*d12;
if den <=0
    return
end;
num=d11*d22-d12.^2;
dtilde=num/den;
lambdatilde=(d11-d12)/den;
if (lambdatilde>0 & lambdatilde<1) & dtilde<d 
    d=dtilde;
    lambda=lambdatilde;
    vert=0;
end;


























