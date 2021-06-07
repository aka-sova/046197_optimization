function [at_Vector,t_Iter,RunTime] = SPGD_EstimatedCoefficients(x,y,NumberOfCoeff,r,a0,b,StopCond,MaxStep)

PowerElement	= 0:NumberOfCoeff-1;
Mat_x           = repmat(x,1,length(PowerElement));
X               = Mat_x.^repmat(PowerElement, length(x), 1);

m           = length(x);
at_Vector   = zeros(NumberOfCoeff,MaxStep);
RunTime     = zeros(1,MaxStep+1);
RandRows    = floor(1+m*rand(1,b));
b_X         = X(RandRows,:);
b_y         = y(RandRows);
t_Iter      = 0 ;
D           = 2*r;
G           = D * max(eig((X.'*X)/m)) + norm(X.'*y/m);
a           = a0; 

gt          = 1/b * b_X.' * (b_X*a-b_y);

while norm(gt)> StopCond && t_Iter < MaxStep
tic
t_Iter      = t_Iter +1 ;
StepSize	= D/(G*sqrt(t_Iter+1));   
a           = ComputeProjectionOnC(a-StepSize*gt,r);

at_Vector(:,t_Iter) = a;
RunTime(t_Iter+1)   = RunTime(t_Iter) + toc;
gt                  = 1/b * b_X.' * (b_X*a-b_y);

end
end

function PC = ComputeProjectionOnC(a,r)
if norm(a)<=r
    PC = a;
else
    PC = r/norm(a) * a;
end
end