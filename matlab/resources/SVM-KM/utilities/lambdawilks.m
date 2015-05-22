function  LW=lambdawilks(x,y)

% Usage
%  LW=lambdawilks(x,y)
%
%  Calculate the Lambda-Wilks value of data x associated with 
%  label y
%  
%  This algo can handle arbitrary number of class but y must
%  coded   in the following way
%  
%   y=i  means x is in class i

if ~isempty(x)
    nbclass=max(y);
    nbdata=size(x,1);
    nbfeat=size(x,2);
    W=zeros(nbfeat,nbfeat);
    %keyboard
    for i=1:nbclass
        ind=find(y==i);
        nbclass_i=length(ind);
        W=W+nbclass_i*cov(x(ind,:),1);
    end;
    W=W/nbdata;
    T=cov(x,1);
    LW=det(W)/det(T);
else
    LW=NaN;
end;



