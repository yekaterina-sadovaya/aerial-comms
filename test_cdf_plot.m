clear;
lam = 0.9;
D = 1;
D_1 = 1;
c = 2;
rho = (lam * D)/c;
rho_1 = lam * D_1;
M = round(0.5*(1 + rho)*c + 10*rho*sqrt(c));
k_max = M/c - 1;


% compute tau (here we solve 2 equations from the book)
nu = optimvar('nu',1,"LowerBound",0, "UpperBound", 1);
eqn1 = lam*D*(1 - 1/nu) + c*log(1/nu) == 0;
eqn2 = rho*(1 - nu) + nu*(log(nu)) == 0;
prob = eqnproblem;
prob.Equations.eqn1 = eqn1;
prob.Equations.eqn2 = eqn2;
x0.nu = 1e-5;
[sol,fval,exitflag] = solve(prob,x0);
nu_val = sol.nu;
tau_val = 1/nu_val;

% compute actual p values
% 1. create a system of M eqations
p = sym('p', [1, M+1]);
e_lambdaD = exp(-lam * D);
for i = 0:M
    p_first = 0;
    for j = 0:c
        if j < M
            p_j = p(j+1);
        else
            p_j = p(M+1)*(tau_val^(-(j-M)));
        end
        p_first = p_first + p_j*((lam*D)^i)*e_lambdaD/factorial(i);
    end
    p_second = 0;
    for j = (c+1):(i+c)
        if j < M
            p_j = p(j+1);
        else
            p_j = p(M+1)*(tau_val^(-(j-M)));
        end
        p_second = p_second +  p_j*((lam*D)^(i+c-j))*e_lambdaD/factorial(i+c-j);
    end
    my_field = strcat('eq',num2str(i+1));
    system.(my_field) = p_first + p_second == p(i+1);
end
eqs = [system.eq1];
for n = 2:(M+1)
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
P = A\B;

% p = zeros(1, M+1);
% p0 = 0.7845;
% for i=1:M
%     p(i) = p0 * 0.5^i;
% end
% P = p/sum(p);
    
q = zeros(1, 2*M+1);
for l =0:2*M
    if l == 0
        q(l+1) = sum(P(1:c));
    else
        if c+l >= M
            q(l+1) = P(M) * tau_val^(-(c + l - M));            
        else
            q(l+1) = P(c+l);
        end
    disp(q(l+1));
    end
end

%x_all = linspace(0, 20, 20);
x_all = 0:20;
n = length(x_all);
yc_value = zeros(1, n);
xc_value = zeros(1, n);
for n_i=1:n
    x = x_all(n_i);
    k = floor(x/D);
    yc_value(n_i) = W_mdc(k, x, c, D, lam, q);
    xc_value(n_i) = x;
end

plot(xc_value, yc_value);


function res = W_mdc(k, x, c, D, lam, q)
    
    temp = 0;
    for j=0:(k * c - 1)
        if (k * c - j - 1) <= length(q)
            Q = sum(q(1:(k * c - j - 1)));
        else
            Q = sum(q);
        end
        temp = temp + Q*exp(lam*(x - k*D))*((- lam*(x - k*D))^j)/ factorial(j);
    end
    res = exp(lam*(x - k*D))*temp;
    
end