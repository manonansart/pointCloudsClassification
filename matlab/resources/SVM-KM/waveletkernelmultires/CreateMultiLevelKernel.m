function [K,Kt]=CreateMultiLevelKernel(xapp,xtest,kerneloption,level)

% USAGE
%
% [K,Kt]=CreateMultiLevelKernel(xapp,xtest,kerneloption,level)
%
%  This function creates multiscale kernels K and Kt for the <xapp,xapp>
%  and <xtest,xapp>
% 
%  Inputs 
%  xapp and xtest are datasets
%  
%  kerneloption is the wavelet kernel configuration see tensorwavkernel
%  
%  level denotes the decomposition of the kernel. level is a 2 dimension
%  kernel with as many rows as desired decomposition.
%  e.g      level=[-3 0; 1 2; 3 7];
%  decomposes a wavelet kernel from -3 to 7 in 3 kernel with the given
%  level decomposition.
%  if 'vect' is given as a parameters in kerneloption then the
%  decomposition is done according to the vectors given in
%  kerneloption.vect
%
%  7/11/2004 A.R


if level(1,1)~=kerneloption.jmin
    error('Erreur de compatibilité entre niveau et noyau');
end;
if level(size(level,1),2)~=kerneloption.jmax
    error('Erreur de compatibilité entre niveau et noyau');
end;

for i=1:size(level,1)
    kerneloption2=kerneloption;
    if isfield(kerneloption,'vect')
        [aux]=ismember(kerneloption.vect(:,1),level(i,1):level(i,2));
        ind=find(aux); 
        kerneloption2.vect=kerneloption.vect(ind,:);
    end
    
    kerneloption2.jmin=level(i,1);
    kerneloption2.jmax=level(i,2);
   
    if i~=1;
        kerneloption2.father='off';
    end;
    %kerneloption2.C=kerneloption.C(ind,:);
    %kerneloption2.D=kerneloption.D(ind,:);
    %kerneloption2.vect
    [K(:,:,i),KernelInfo]=tensorwavkernel(xapp,xapp,kerneloption2);
    if ~isempty(xtest)
     [Kt(:,:,i),KernelInfo]=tensorwavkernel(xtest,xapp,kerneloption2);
 else
     Kt=[];
    end;    
 end;