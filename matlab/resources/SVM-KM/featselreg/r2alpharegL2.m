function bound=r2alpharegL2(xapp,kernel,kerneloption,C,alpha,epsilon);


 [beta,r2,pos]= r2smallestsphere(xapp,kernel,kerneloption,C);
 bound=4*r2*sum(alpha)+size(xapp,1)*epsilon;