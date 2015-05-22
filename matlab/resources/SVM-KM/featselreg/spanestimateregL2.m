function Bound=spanestimateregL2(alpha,K,n,C,epsilon)

eta=0.00001;
lambdaregul=1e-7;
nbsvaux=length(alpha);
AlphaStarTemp=alpha(1:n);
AlphaTemp=alpha(n+1:end);
newpos=find(alpha(1:n)>0|alpha(n+1:2*n)> 0);
k=K(newpos,newpos) + 1/C*eye(length(newpos));
D=(eta./(AlphaTemp(newpos)+AlphaStarTemp(newpos)));
ksvaux=[k ones(length(newpos),1)];
ksvaux=[ksvaux; [ones(1,length(newpos)) 0]];
ksvaux=ksvaux+diag([D;0]);
sp2aux=1./diag(inv(ksvaux+lambdaregul*eye(size(ksvaux))));
sp2aux=sp2aux(1:length(newpos))-D;
Bound=(AlphaStarTemp(newpos)+AlphaTemp(newpos))'*sp2aux+n*epsilon;