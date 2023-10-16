clear; 

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
lambda_HAPS = 10;

lambda = lambda_HAPS;
D = 0.5;

% first compute the parameters for Q computation 
c = 15;
x = linspace(0.1, 0.5, 100);
x_i = x(1);
k = floor(x_i/D);
m = k*c - 1;
e_lambdaD = exp(-lambda * D);

% compute maximum number of i
rho = (lambda*D)/c;
M = 0.5*(1 + rho)*c + 10*rho*sqrt(c);
M = floor(M);
if rho >= 1
    disp('Warning! Method works only for rho < 1.')
end

% compute tau 
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

% compute p
p = sym('p', [1, M]);
for i = 1:M
    p_first = 0;
    for j = 1:c
        if j <= M
            p_j = p(j);
        else
            p_j = p(M)*tau_val^(-(j-M));
        end
        p_first = p_first + p_j*((lambda*D)^i)*e_lambdaD/factorial(i);
    end
    p_second = 0;
    for j = (c+1):(i+c)
        if j <= M
            p_j = p(j);
        else
            p_j = p(M)*tau_val^(-(j-M));
        end
        p_second = p_second +  p_j*((lambda*D)^(i+c-j))*e_lambdaD/factorial(i+c-j);
    end
    my_field = strcat('eq',num2str(i));
    system.(my_field) = p_first + p_second == p(i);
end
eqs = [system.eq1];
for n = 2:M
    my_field = strcat('eq',num2str(n));
    eqs = [eqs, system.(my_field)];
end
norm_eq = sum(p) == 1;
eqs = [eqs, norm_eq];
[A, B] = equationsToMatrix(eqs);
A = double(A);
B = double(B);
P_ALL = linsolve(A,B)

