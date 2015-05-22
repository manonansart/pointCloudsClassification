function [kernel,KernelInfo]=tensorwavkernel(x,xsup,kerneloption)


% USAGE
% 
%  [kernel,KernelInfo]=tensorwavkernel(x,xsup,kerneloption)
%
%  Process a tensor product wavelet kernel. 
%  the data needs not to lies in [0,1]^d
%
%  at this time, this kernel needs the Wavelab Toolbox
%  developed by Donoho et al. at Stanford University.
%
%  INPUT
%
%  x and xsup   :  data points entry
%  kerneloption :  option of the wavelet kernel. this is a struct
%                  containing the following field
%
%                   wname : the wavelet  'Haar','Daubechies', 'Symmlet'
%                   par   : length of wavelet or number of vanishing moments
%                   pow : nb of points in the wavelet
%                   jmax : maximal resolution
%                   jmin : minimal resolution
%                   father : 'on'/'off'   do not include tha scaling function in the span.
%                   crossterm : 'off'/'on' ('off' default)  
%                   coeffj: coefficients that weight each wavelet and that
%                   is powered to j  (coeffj^j)
%            
%                   default : 'Symmlet' pow=2^10 jmax=4 jmin=0
%                   check : 0 (default) or 1  % check the positivity of a square Gram matrix 
%
%   The equation for the 1-dimension kernel is
%    
%    kernelaux=[C*fatherx]'*[C*fatherxsup] + [D*fx]'*[D*fxsup];
%
%  for father='off', the scaling part is not included in the kernel 
%  for crossterm='on', the crossterm part is added to the kernel

%           01/04/2002 Alain Rakotomamonjy

%
% the usual checking of input entry
%
if nargin <3
    kerneloption.wname='Symmlet';
    kerneloption.par=4;
    kerneloption.pow=10;  
    kerneloption.jmax=4;
    kerneloption.jmin=0;
end;

if ~isfield(kerneloption,'wname')
    kerneloption.wname='Symmlet';
end;
if ~isfield(kerneloption,'par')
    kerneloption.par=4;
end;

if ~isfield(kerneloption,'pow')
    kerneloption.pow=10;
end;

if ~isfield(kerneloption,'jmax')
    kerneloption.jmax=4;
end;

if ~isfield(kerneloption,'jmin')
    kerneloption.jmin=0;
end;
if ~isfield(kerneloption,'check')
    kerneloption.check=0;
end;
if ~isfield(kerneloption,'crossterm')
    kerneloption.crossterm='off';
end;
if ~isfield(kerneloption,'coeffj')
    kerneloption.coeffj=1/sqrt(2);
end;
if ~isfield(kerneloption,'father')
    kerneloption.father='on';
end;

[phi,psi,xval]=waveletfunction(kerneloption.wname,kerneloption.par,2^(kerneloption.pow));



xmin=min(xval);
xmax=max(xval);



dim=size(x,2);
N=size(x,1);
Nxsup=size(xsup,1);
kernel=ones(N,Nxsup);






mint=0;   %   This is the support of data point. here we work on the interval [0,1]^d
maxt=1;   %   if we are not on the interval, one should look for
mint=min(min([x ; xsup]));   
maxt=max(max([x; xsup]));  
if ~isfield(kerneloption,'vect')
    vect=[];
    for j=kerneloption.jmin:kerneloption.jmax        
        aux=[floor(2^j*mint-xmax)-1:1:floor(2^j*maxt-xmin)+1]';
        vect=[vect;[ones(length(aux),1)*j aux]];
        
    end;
else
    if min(kerneloption.vect(:,1))==kerneloption.jmin & max(kerneloption.vect(:,1))==kerneloption.jmax
        vect=kerneloption.vect;
    else
        error('There is an imcompatibility between ''vect'' and ''jmin'' and ''jmax''');
    end;
end; 

Nvect=length(vect);
for idim=1:dim
    iter=1;
    
    %  fx=[];
    %  fxsup=[];
    %  fatherx=[];
    %  fatherxsup=[];
    fx=zeros(Nvect,N);
    fxsup=zeros(Nvect,Nxsup);
    
    t=x(:,idim);
    tsup=xsup(:,idim);
    
    
    for iter=1:Nvect
        jaux=vect(iter,1);
        kaux=vect(iter,2);
        
        xvaldt=2^(-jaux)*(xval+kaux);  % support de l'ondelette jaux, kaux
        xvaldtmin=min(xvaldt); 
        xvaldtmax=max(xvaldt); 
        
        %
        %   treating x 
        %
        indice_t_in_support=find(t >= xvaldtmin & t <=xvaldtmax);
        ti=t(indice_t_in_support);
        if length(ti)>0
            dist=abs(ones(length(ti),1)*xvaldt- ti*ones(1,length(xvaldt)));
            [value,indice]=min(dist');
            fx(iter,indice_t_in_support)=psi(indice);
        else
            
            fx(iter,:)=zeros(1,length(t));
        end;
        
        %
        %   treating xsup 
        %
        
        indice_t_in_support=find(tsup >= xvaldtmin & tsup <=xvaldtmax);
        ti=tsup(indice_t_in_support);
        if length(ti)>0
            dist=abs(ones(length(ti),1)*xvaldt- ti*ones(1,length(xvaldt)));
            [value,indice]=min(dist');
            fxsup(iter,indice_t_in_support)=psi(indice);
        else
            fxsup(iter,:)=zeros(1,length(tsup));
        end;
    end
    
    
    
    %---------------------------------------------------------
    %
    %    working on scaling function
    %
    %---------------------------------------------------------
    
    
    
    indvectlow=find(vect(:,1)==kerneloption.jmin);
    Nvectlow=length(indvectlow);   
    
    
    fatherx=zeros(Nvectlow,N);
    fatherxsup=zeros(Nvectlow,Nxsup);
    if strcmp(kerneloption.father,'on');   % take into account scaling function
        for i=1:Nvectlow;
            jaux=vect(indvectlow(i),1);
            kaux=vect(indvectlow(i),2);
            xvaldtmin=2^(-jaux)*(xmin+kaux);  % looking for the lower bound of the dilatotranslated wavelet
            
            xvaldt=2^(-jaux)*(xval+kaux);  % support de l'ondelette jaux, kaux
            xvaldtmin=min(xvaldt); 
            xvaldtmax=max(xvaldt); 
            
            indice_t_in_support=find(t >= xvaldtmin & t <=xvaldtmax);
            ti=t(indice_t_in_support);
            if length(ti)>0
                dist=abs(ones(length(ti),1)*xvaldt- ti*ones(1,length(xvaldt)));
                [value,indice]=min(dist');
                fatherx(i,indice_t_in_support)=phi(indice);
            else
                fatherx(i,:)=zeros(1,length(t));
            end;
            
            indice_t_in_support=find(tsup >= xvaldtmin & tsup <=xvaldtmax);
            ti=tsup(indice_t_in_support);
            if length(ti)>0
                dist=abs(ones(length(ti),1)*xvaldt- ti*ones(1,length(xvaldt)));
                [value,indice]=min(dist');
                fatherxsup(i,indice_t_in_support)=phi(indice);
            else
                fatherxsup(i,:)=zeros(1,length(tsup));
            end;
        end;
    end;
    
    % Calculating coefficient matrices C and D    
    if ~isfield(kerneloption,'D') & ~isfield(kerneloption,'C') & ~isfield(kerneloption,'Nfunct')
        Nfunct=Nvect;
        D=zeros(Nfunct,Nvect);
        C=zeros(Nfunct,Nvectlow);
        
        
        for iter=1:Nvect;
            D(iter,iter)=kerneloption.coeffj^vect(iter,1);
            if iter <= Nvectlow
                C(iter,iter)=kerneloption.coeffj^vect(iter,1);
            end;
        end;
    end;
    
    if isfield(kerneloption,'D') | isfield(kerneloption,'C')
        [NbligC,NbcolC,DimC]=size(kerneloption.C);
        if DimC==dim 
            C=kerneloption.C(:,:,idim);
        else
            C=kerneloption.C(:,:,1);
        end
        [NbligD,NbcolD,DimD]=size(kerneloption.D);
        if DimD==dim 
            D=kerneloption.D(:,:,idim);
        else
            D=kerneloption.D(:,:,1);
        end
        
        
        if NbcolC > Nvectlow
            C=C(:,1:Nvectlow);
        else
            C=[C zeros(size(C,1), Nvectlow-NbcolC)];
        end;
        
        if NbcolD > Nvect
            D=D(:,1:Nvect);
        else
            D=[D zeros(size(D,1), Nvect-NbcolD)];
        end;
        
        
    end;
    
    % Calculting Kernel for dimension I
    
    kernelaux=+ [D*fx]'*[D*fxsup];
    
    if strcmp(kerneloption.father,'on');
        kernelaux=kernelaux+ [C*fatherx]'*[C*fatherxsup] ;
    end;
    
    if strcmp(kerneloption.crossterm,'on');
        kernelaux=kernelaux+ [C*fatherx]'*[D*fxsup]+[D*fx]'*[C*fatherxsup] ;
    end;
    
    % updating kernel
    
    
    kernel=kernel.*kernelaux;
    kerneldim(:,:,idim)=kernelaux;
    FatherxMat(:,:,idim)=fatherx;
    FatherxsupMat(:,:,idim)=fatherxsup;
    MotherxMat(:,:,idim)=fx;
    MotherxsupMat(:,:,idim)=fxsup;
    CMat(:,:,idim)=C;
    DMat(:,:,idim)=D;
end;

if kerneloption.check & Nxsup==N
    [v,d]=eig(kernel);
    d=diag(d);
    
    if min((d+1E-10))>0
        fprintf('Positive definite Sanity Check : OK\n');
    end;
end;


KernelInfo.Fatherx=FatherxMat;
KernelInfo.Fatherxsup=FatherxsupMat;
KernelInfo.Motherx=MotherxMat;
KernelInfo.Motherxsup=MotherxsupMat;
KernelInfo.vect=vect;
KernelInfo.C=CMat;
KernelInfo.D=DMat;
KernelInfo.Kdim=kerneldim;