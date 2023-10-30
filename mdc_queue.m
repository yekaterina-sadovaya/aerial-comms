clear; 

%% Setup modeling parameters
r = 10;
t_star = 1/r;
C = 60;
C_UE = 200;
C_UAV = 400;
C_HAP = 1000;
D_UE = C/C_UE;
D_UAV = C/C_UAV;
D_HAPS = C/C_HAP;
lambda_UAV = 0.5;
lambda_HAPS = 0.9;

lambda = lambda_HAPS;
D = 1;

c = 1;
n_points = 20;

% Set range for time
% x should start from D decause we can't have waiting time less then 
% processing time
% also (k-1)D <= x < kD
x = linspace(0, 20, n_points);

% Compute e^(-lambda*D)
e_lambdaD = exp(-lambda * D);

%% P parameters computation
% compute maximum number equations (i)
rho = (lambda*D)/c;
M1 = 0.5*(1 + rho)*c + 10*rho*sqrt(c);
M1 = round(M1);
%M = max(M1, c*5);
M = M1;

% check if all conditions are satisfied
if rho >= 1
    disp('Warning! Method works only for rho < 1.')
end

if lambda*D ~= c*rho
    disp('Warning! lambda*D != c*rho')
end

% compute tau (here we solve 2 equations from the book)
nu = optimvar('nu',1,"LowerBound",0, "UpperBound", 1);
eqn1 = lambda*D*(1 - 1/nu) + c*log(1/nu) == 0;
eqn2 = rho*(1 - nu) + nu*(log(nu)) == 0;
prob = eqnproblem;
prob.Equations.eqn1 = eqn1;
prob.Equations.eqn2 = eqn2;
x0.nu = 0.5;
[sol,fval,exitflag] = solve(prob,x0);
nu_val = sol.nu;
tau_val = 1/nu_val;

% compute actual p values
% 1. create a system of M eqations
p = sym('p', [1, M]);
for i = 0:(M-1)
    p_first = 0;
    for j = 0:(c-1)
        if j < M
            p_j = p(j+1);
        else
            p_j = p(M)*tau_val^(-(j-M));
        end
        p_first = p_first + p_j*((lambda*D)^i)*e_lambdaD/factorial(i);
    end
    p_second = 0;
    for j = (c+1):(i+c)
        if j < M
            p_j = p(j+1);
        else
            p_j = p(M)*tau_val^(-(j-M));
        end
        p_second = p_second +  p_j*((lambda*D)^(i+c-j))*e_lambdaD/factorial(i+c-j);
    end
    my_field = strcat('eq',num2str(i+1));
    system.(my_field) = p_first + p_second == p(i+1);
end
eqs = [system.eq1];
for n = 2:M
    my_field = strcat('eq',num2str(n));
    eqs = [eqs, system.(my_field)];
end
% 2. add normalization equation
norm_eq = sum(p) == 1;
eqs = [eqs, norm_eq];
% 3. convert equation coefficients to numerical matrices
[A, B] = equationsToMatrix(eqs);
A = double(A);
B = double(B);
% 4. solve for p
% P_ALL = linsolve(A,B);
P_ALL = A\B;
%P_ALL = sort(P_ALL,'descend');

p = zeros(1, M+1);
q = zeros(1, 2*M+1);
p0 = 0.7845;
for i=1:M
    p(i) = p0 * 0.5^i;
end
P = p/sum(p);
P_ALL = P;

%% computation of Q values and CDF function F
q_all = comp_Q(M, tau_val, c, P_ALL);
F_CDF = zeros(1, length(x));
for i_point = 1:length(x)
    x_i = x(i_point);
    k = floor(x_i/D);
    if k > 0        
        F_CDF(i_point) = W_mdc(k, x_i, c, D, lambda, q_all);
    else
        F_CDF(i_point) = 0;
    end
end


%% plotting
plot(x, F_CDF);

% first compute Q 
function q_all = comp_Q(M, tau, c, P_ALL)
    q_all = zeros(1,2*M);
    for j = 0:2*M
        if j == 0
            q_curr = sum(P_ALL(1:c));
        else
            if (c+j) >= M
                if (M+1) <= length(P_ALL)
                    q_curr = P_ALL(M+1)* tau^(c + j - M);
                else
                    q_curr = 0;
                end
            else
               q_curr = P_ALL(c+j+1);
            end
        end
        q_all(j+1) = q_curr;
    end
end

function res = W_mdc(k, x, c, D, lam, q)
    
    temp = 0;
    for j=0:(k * c - 1)
        Q = sum(q(1:(k * c - j - 1)));
        temp = temp + Q*exp(lam*(x - k*D))*((- lam*(x - k*D))^j)/ factorial(j);
    end
    res = exp(lam*(x - k*D))*temp;
    
end