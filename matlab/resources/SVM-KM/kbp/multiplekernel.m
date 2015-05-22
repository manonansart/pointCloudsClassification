function K=multiplekernel(x,kernel,kerneloptionvec,xapp,solution);

%
% USAGE
%
% K=multiplekernel(x,kernel,kerneloptionvec,xapp,solution);
%
% process the kernel K(x,xapp). If solution is provided, it process
% only the kernel of the Support Vector xapp 
%
%


K=[];
nbkerneloption=length(kerneloptionvec);
if nargin < 4
    for i=1:nbkerneloption
        kerneloption=kerneloptionvec(i);
        K=[K svmkernel(x,kernel,kerneloption)];
    end;
else
    nbapp=size(xapp,1);
    if nargin < 5
        indxsup=1:nbapp*nbkerneloption;    
    else
        indxsup=solution.indxsup;
    end;
    
    K=zeros(size(x,1),length(indxsup));
    for i=1:nbkerneloption
        kerneloption=kerneloptionvec(i);
        indice=find(indxsup <= i*nbapp & indxsup > (i-1)*nbapp);    % les points qui sont concernes par ce kerneloption
        if ~isempty(indice)
            Kaux=svmkernel(x,kernel,kerneloption, xapp(indxsup(indice)-(i-1)*nbapp,:));
            K(:,indice)=Kaux;
        end
        
    end;
end

