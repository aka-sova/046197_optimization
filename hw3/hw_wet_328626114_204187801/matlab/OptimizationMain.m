%% Main
close all; 
clear; 
clc; 
%% Part 1
m   = 1E4;
n   = 3;
x   = -1 + 2*rand(m,1);

a0  = 0;
a1  = 1;
a2  = 0.5;
a3  = -2;


a   = [a0;a1;a2;a3];
y   = polyval(a(end:-1:1),x) + sqrt(0.5)*randn(size(x,1),1);

Estimated_a = ComputeEstimatedCoefficients(x,y,n+1);

t=linspace(-1,1,m);
figure(1);
fig1_Leg = [];
plot(t,polyval(a(end:-1:1),t),'LineWidth',4,'Color','b'); hold on ;
fig1_Leg = [fig1_Leg "Actual Polynom"];
scatter(x,y,25,'g')
fig1_Leg = [fig1_Leg "y Points"];
plot(t,polyval(Estimated_a(end:-1:1),t),'LineWidth',2,'Color','r');
fig1_Leg = [fig1_Leg "Estimated Polynom"];
legend(fig1_Leg);
PowerElement    = 0:n;
Mat_x           = repmat(x,1,length(PowerElement));
X               = Mat_x.^repmat(PowerElement, length(x), 1);

h_opt = 1/(2*m) * norm(y-X*Estimated_a)^2;

%% Part 2
r =4;
a0 = randn(n+1,1);
StopCond = 1E-10;

while norm(a0) > r
    a0 = randn(n+1,1);
end

MaxStep = 2000;
L = max(eig((X.'*X)/m));
fig2_Leg = ["Dec Step" "AdaGrad"];
fig3_Leg = ["1/10L" "1/L" "10/L"];
fig4_Leg = ["AdaGrad" "1/L"];
figure(4) % Figure AdaGrad vs Const
figure(3) % Figure 3 Const Step
figure(2) % Figure Dec vs AdaGrad

for StepSizeType=1:3
    NumberOfForRuns = 1;
    if StepSizeType == 3
        figure(3)
        NumberOfForRuns = 3;
    end
    for ii = 1:NumberOfForRuns
        ConstStepSize = 1/(10*L)*(10^(ii-1));
        [at_Vector,t] = PGD_EstimatedCoefficients(x,y,n+1,r,a0,StepSizeType,StopCond,MaxStep,ConstStepSize);
        PGD_at_Vector = [a0 at_Vector(:,1:t)];
        t_Iter = 0:t;
        ht = 1/(2*m) * vecnorm(y-X*PGD_at_Vector).^2;
        PGD_Error = ht - h_opt;
        semilogy(t_Iter,PGD_Error);
        
        if StepSizeType == 2 ||  (StepSizeType == 3 && ii==2)
            figure(4);
            semilogy(t_Iter,PGD_Error);
            if StepSizeType == 3
                figure(3)
            end
        end
        %         legend(num2str(StepSizeType));
        grid on; hold on;
    end
end

figure(4) % Figure AdaGrad vs Const
legend(fig4_Leg);
figure(3) % Figure 3 Const Step
legend(fig3_Leg);
figure(2) % Figure Dec vs AdaGrad
legend(fig2_Leg);

%% Part 3
SPGD_Leg = [""];
figure(5) % Figure Error vs iter
figure(6) % Figure Error vs iterTime
for ii = 1:5
    b=10^(ii-1);
    SPGD_Leg(ii) = ['b = ' num2str(b)];
    [at_Vector,t,RunTimeVec] = SPGD_EstimatedCoefficients(x,y,n+1,r,a0,b,StopCond,MaxStep);
    SPGD_at_Vector = [a0 at_Vector(:,1:t)];
    t_Iter = 0:t;
    ht = 1/(2*m) * vecnorm(y-X*SPGD_at_Vector).^2;
    SPGD_Error = ht - h_opt;
    figure(5) % Figure Error vs iter
    semilogy(t_Iter,SPGD_Error);
    grid on; hold on;
    figure(6) % Figure Error vs iterTime
    semilogy(RunTimeVec(1:t+1),SPGD_Error);
    grid on; hold on;
end
legend(SPGD_Leg)