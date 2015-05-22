function [X]=fileaccess(datafile,Indice,Dimension,fid,Format,Offset,CaracEntreExemple)
%
%
% USAGE
% 
% [X]=fileaccess(datafile,Indice,Dimension,fid)
%
% default
%
% Format=16;
% Offset=0;
% CaracEntreExemple=2;
%
% This function is able to read ASCII File saved under windows text format


if nargin <4
    Format=16;
    Offset=0;
    CaracEntreExemple=2;
end;


N=length(Indice);
X=zeros(N,Dimension);
fid=fopen(datafile,'r+');
if fid>1
  %  keyboard
    for i=1:N
        
     %   if Dimension~=1;
            if fseek(fid,(Dimension*Format+CaracEntreExemple)*(Indice(i)-1)-Offset,-1)==0
                
                X(i,:)=([fscanf(fid,'%f',Dimension)])';
            end;
%         else
%             if fseek(fid,(Format)*(Indice(i)-1)-Offset,-1)==0
%                 
%                 X(i,:)=([fscanf(fid,'%f',Dimension)])';
%             end;
%         end;
%         
    end;
    fclose(fid) ;
end;
