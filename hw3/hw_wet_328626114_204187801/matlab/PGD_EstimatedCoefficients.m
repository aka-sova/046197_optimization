function [at_Vector,t] = PGD_EstimatedCoefficients(x,y,NumberOfCoeff,r,a0,StepSizeType,StopCond,MaxStep,ConstStepSize)

PowerElement	= 0:NumberOfCoeff-1;
Mat_x           = repmat(x,1,length(PowerElement));
X               = Mat_x.^repmat(PowerElement, length(x), 1);

SumSquaredGrad	= 0;

m           = length(x);
at_Vector	= zeros(NumberOfCoeff,MaxStep);
t           = 0 ;
D           = 2*r;
a           = a0; 

G           = D * max(eig((X.'*X)/m)) + norm(X.'*y/m);
gt          = 1/m * X.' * (X*a-y);



while norm(gt)> StopCond && t < MaxStep
t = t +1 ;

switch StepSizeType
    case 1
        StepSize        = D/(G*sqrt(t));       
    case 2
        SumSquaredGrad	= SumSquaredGrad + norm(gt)^2;
        StepSize        = D/(sqrt(2*SumSquaredGrad));
    case 3
        StepSize        = ConstStepSize; 
end

a               = ComputeProjectionOnC(a-StepSize*gt,r);
at_Vector(:,t)  = a;
gt              = 1/m * X.' * (X*a-y);

end
end

function PC = ComputeProjectionOnC(a,r)
if norm(a)<=r
    PC = a;
else
    PC = r/norm(a) * a;
end
end