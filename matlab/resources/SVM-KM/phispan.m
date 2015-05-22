function T=phispan(x,phitype,dilation,uo,a,bord1,bord2,Family,par);

% USAGE T=phispan(x,phitype,dilation,uo,a,bord1,bord2) 
% 
% This function returns a matrix Tij that contains elements
% \phi_j(x_i). The size of the matrix is N*M where N is the number
% of data x and M the number of parametric functions.
%
% phitype   is a string containing the type of parametric functions desired.
%           'sin', 'sin2','unite','wavelet',sin2D    
%
% dilation, uo,a, bord1, bord2 are parameters that should be used for wavelet span
%
%
% Octobre 2000 - AR

switch lower(phitype)
case 'sin'
    phi0=ones(size(x));
    phi1=sin(x);
    phi2=cos(x);
    T=[phi0 phi1 phi2];
case  'sin2'
    phi0=ones(size(x));
    phi3=sin(x);
    phi4=cos(x);
    phi1=sin(2*x);
    phi2=cos(2*x);
    T=[phi0 phi1 phi2 phi3 phi4];
    
case 'unite'
    phi0=ones(size(x));
    T=phi0;
case 'sinc_sin'
    phi0=ones(size(x));
    phi3=sin(x);
    phi4=cos(x);
    phi5=sinc(x-1);
    phi6=sinc(x-2);	
    phi7=sinc(x-3);
    phi8=sinc(x-4);
    phi9=sinc(x-5);
    phi10=sinc(x-6);
    phi11=sinc(x-7);
    phi12=sinc(x-8);
    phi13=sinc(x-9);
    T=[phi0 phi3 phi4 phi5 phi6 phi7 phi8 phi9 phi10 phi11 phi12 phi13];
case 'sinc13_sin'
    phi0=ones(size(x));
    phi3=sin(x);
    phi4=cos(x);
    phi5=sin(1*pi*(x-1))./(1*pi*(x-1));
    phi6=sin(1*pi*(x-2))./(1*pi*(x-2));	
    phi7=sin(1*pi*(x-3))./(1*pi*(x-3));
    phi8=sin(1*pi*(x-4))./(1*pi*(x-4));
    phi9=sin(1*pi*(x-5))./(1*pi*(x-5));
    phi10=sin(1*pi*(x-6))./(1*pi*(x-6));
    phi11=sin(1*pi*(x-7))./(1*pi*(x-7));
    phi12=sin(1*pi*(x-8))./(1*pi*(x-8));
    phi13=sin(1*pi*(x-9))./(1*pi*(x-9));
    phi14=sin(3*pi*(x-1))./(3*pi*(x-1));
    phi15=sin(3*pi*(x-2))./(3*pi*(x-2));
    phi16=sin(3*pi*(x-3))./(3*pi*(x-3));
    phi17=sin(3*pi*(x-4))./(3*pi*(x-4));
    phi18=sin(3*pi*(x-5))./(3*pi*(x-5));
    phi19=sin(3*pi*(x-6))./(3*pi*(x-6));
    phi20=sin(3*pi*(x-7))./(3*pi*(x-7));
    phi21=sin(3*pi*(x-8))./(3*pi*(x-8));
    phi22=sin(3*pi*(x-9))./(3*pi*(x-9));
    T=[phi0 phi3 phi4 phi5 phi6 phi7 phi8 phi9 phi10 phi11 phi12 phi13 phi14 phi15 phi16 phi17 phi18 phi19 phi20 phi21 phi22];
case 'sinc13'
    phi4=sin(2*x);
    phi5=sin(1*pi*(x-1))./(1*pi*(x-1));
    phi6=sin(1*pi*(x-2))./(1*pi*(x-2));	
    phi7=sin(1*pi*(x-3))./(1*pi*(x-3));
    phi8=sin(1*pi*(x-4))./(1*pi*(x-4));
    phi9=sin(1*pi*(x-5))./(1*pi*(x-5));
    phi10=sin(1*pi*(x-6))./(1*pi*(x-6));
    phi11=sin(1*pi*(x-7))./(1*pi*(x-7));
    phi12=sin(1*pi*(x-8))./(1*pi*(x-8));
    phi13=sin(1*pi*(x-9))./(1*pi*(x-9));
    phi14=sin(3*pi*(x-1))./(3*pi*(x-1));
    phi15=sin(3*pi*(x-2))./(3*pi*(x-2));
    phi16=sin(3*pi*(x-3))./(3*pi*(x-3));
    phi17=sin(3*pi*(x-4))./(3*pi*(x-4));
    phi18=sin(3*pi*(x-5))./(3*pi*(x-5));
    phi19=sin(3*pi*(x-6))./(3*pi*(x-6));
    phi20=sin(3*pi*(x-7))./(3*pi*(x-7));
    phi21=sin(3*pi*(x-8))./(3*pi*(x-8));
    phi22=sin(3*pi*(x-9))./(3*pi*(x-9));
    T=[phi5 phi6 phi7 phi8 phi9 phi10 phi11 phi12 phi13 phi14 phi15 phi16 phi17 phi18 phi19 phi20 phi21 phi22];
    
case 'wavelet';
    T=waveletspan(x,dilation,uo,a,bord1,bord2);
    
case 'orthowavelet';
    i=1;
    nx=length(x);   
    n=1024;
    vector=linspace(0,1,n);
    
    for p=1:length(dilation)
        for k=1:2^dilation(p)
            aux=makewavelet(dilation(p),k,Family,par,'Father',n);
            for m=1:nx;
                T(m,i)=valinterp(aux,vector,x(m));
            end;
            
            i=i+1;
        end;
    end;
    
case 'sin2d';
        phi0=ones(length(x),1);
    T=[phi0 sin(x(:,1)).*sin(x(:,2))];
    
end;
