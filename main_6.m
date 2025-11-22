clc; clear all; close all;

params.m=4.34;
params.J=diag([0.0820;0.0845;0.1377]);
params.g=9.81;
ts=0.01;
dt=0.001;
steps=round(ts/dt);
T=0.1;
Ntraj=100;

X_list={};
Xp_list={};
U_list={};

for tr=1:Ntraj
    % initial condition
    p0=zeros(3,1);
    v0=zeros(3,1);
    R0=eye(3);
    omega0=zeros(3,1);
    x=[p0;v0;R0(:); omega0];

    
    mu_u=[0;0;0;0];
    Sigma_u=diag([2;2;2;2]);
    u=mvnrnd(mu_u, Sigma_u).';
    u(1) = u(1)+params.m*params.g;

    Nsteps=round(T/ts);
    for k=1:Nsteps
        x_k=x;
        for j=1:steps
            x = rk4_step(x, u, params, dt);
        end
        x_kp1=x;
        X_list{end+1}=x_k;
        Xp_list{end+1}=x_kp1;
        U_list{end+1}=u;
    end
   
end

% convert to matrices
X   = cell2mat(X_list);      % size n x K
Xp  = cell2mat(Xp_list);     % size n x K
U   = cell2mat(U_list);      % size 4 x K
   
% for all snapshots, compute the lifted data
p=3;
K=size(X,2);
n = size(X,1);

Z=zeros(n+9*p, K);
Zp=zeros(n+9*p, K);

for k=1:K
    Z(:, k)=observable_phi(X(:, k), p);
    Zp(:,k)=observable_phi(Xp(:,k), p);
end

% implementing EDMD

ZU = [Z; U];  
Nz  = n + 9*p;

lambda = 1e-6;   % regularization (tweak if needed)
G = Zp * ZU' / (ZU*ZU' + lambda*eye(size(ZU,1)));
A = G(:,1:Nz);
B = G(:,Nz+1:end);

C = [eye(n), zeros(n, 9*p)];   % n x Nz



%% generating 50 new trajectories for validation
Nval_traj=50;
Tval=0.1;
Nsteps=round(Tval/ts);
mu_val=[0;0;0;0];
Sigma_val=diag([4;4;4;4]);



% --- error accumulators (plain RMSE) ---
err_p  = [];
err_v  = [];
err_R  = [];
err_w  = [];
err_th = [];

% --- accumulators for normalized RMSE ---
num_p  = 0;  den_p  = 0;
num_v  = 0;  den_v  = 0;
num_R  = 0;  den_R  = 0;
num_w  = 0;  den_w  = 0;
num_th = 0;  den_th = 0;

for tr = 1:Nval_traj
    p0 = zeros(3,1);
    v0 = zeros(3,1);
    R0 = eye(3);
    w0 = zeros(3,1);
    x_true = [p0; v0; R0(:); w0];

    % corresponding lifted state
    z = observable_phi(x_true,p);

    % random constant input for this trajectory
    u = mvnrnd(mu_val, Sigma_val).';
    % u(1) = max(u(1), 1.2*params.m*params.g);
    u(1) = u(1)+params.m*params.g;
    for k = 1:Nsteps
        % --- true nonlinear propagation over one sample period ---
        x = x_true;
        for j = 1:steps
            x = rk4_step(x, u, params, dt);
        end
        x_true = x;

        % --- Koopman linear prediction ---
        z      = A*z + B*u;
        x_pred = C*z;

        % unpack true
        p_true = x_true(1:3);
        v_true = x_true(4:6);
        R_true = reshape(x_true(7:15),3,3);
        w_true = x_true(16:18);

        % unpack predicted
        p_hat  = x_pred(1:3);
        v_hat  = x_pred(4:6);
        R_hat  = reshape(x_pred(7:15),3,3);
        w_hat  = x_pred(16:18);

        % orientation vectors theta_true, theta_hat
        Theta_true = so3_log(R_true);
        Theta_hat  = so3_log(R_hat);
        theta_true = vee(Theta_true);
        theta_hat  = vee(Theta_hat);

        % --- plain errors (for RMSE) ---
        e_p  = norm(p_hat  - p_true);
        e_v  = norm(v_hat  - v_true);
        e_R  = norm(R_hat  - R_true,'fro');
        e_w  = norm(w_hat  - w_true);
        e_th = norm(theta_hat - theta_true);

        err_p(end+1,1)  = e_p;
        err_v(end+1,1)  = e_v;
        err_R(end+1,1)  = e_R;
        err_w(end+1,1)  = e_w;
        err_th(end+1,1) = e_th;

        % --- accumulators for normalized RMSE ---
        num_p  = num_p  + e_p^2;
        den_p  = den_p  + norm(p_true)^2;

        num_v  = num_v  + e_v^2;
        den_v  = den_v  + norm(v_true)^2;

        num_R  = num_R  + e_R^2;
        den_R  = den_R  + norm(R_true,'fro')^2;

        num_w  = num_w  + e_w^2;
        den_w  = den_w  + norm(w_true)^2;

        num_th = num_th + e_th^2;
        den_th = den_th + norm(theta_true)^2;
    end
end

% --- plain RMSEs ---
RMSE_p     = sqrt(mean(err_p.^2));
RMSE_v     = sqrt(mean(err_v.^2));
RMSE_R     = sqrt(mean(err_R.^2));
RMSE_w     = sqrt(mean(err_w.^2));
RMSE_theta = sqrt(mean(err_th.^2));

% --- normalized RMSEs in percent ---
eps_den = 1e-8;
nRMSE_p     = 100*sqrt(num_p  / max(den_p, eps_den));
nRMSE_v     = 100*sqrt(num_v  / max(den_v, eps_den));
nRMSE_R     = 100*sqrt(num_R  / den_R);
nRMSE_w     = 100*sqrt(num_w  / den_w);
nRMSE_theta = 100*sqrt(num_th / den_th);

fprintf("position RMSE = %.4f%%\n", nRMSE_p);
fprintf("velocity RMSE = %.4f%%\n", nRMSE_v);
fprintf("theta RMSE = %.4f%%\n", nRMSE_theta);
fprintf("omega RMSE = %.4f%%\n", nRMSE_w);

% MPC implementation

Nh=10;
t_sim=1.2;Ts=0.01;
Nsim=round(t_sim/Ts);
p_obs=3;
steps = round(Ts / dt);

Nz = size(A,1);     % lifted dimension
n  = size(C,1);     % original state dimension
nu = size(B,2);     % number of control inputs 
Aqp = zeros(Nz*Nh, Nz);
Bqp = zeros(Nz*Nh, nu*Nh);

A_power = eye(Nz);
for i = 1:Nh
    A_power = A_power * A;              % A^i
    Aqp((i-1)*Nz+1:i*Nz, :) = A_power;
    for j = 1:i
        % contribution of u_{j-1} to z_i: A^{i-j} * B
        Aij = A^(i-j) * B;
        Bqp((i-1)*Nz+1:i*Nz, (j-1)*nu+1:j*nu) = Aij;
    end
end


% weighing matrices
w_p = 1000;
w_v = 20;
w_theta = 2000;
w_w = 10;

Qx_pos   = w_p * eye(3);
Qx_vel   = w_v * eye(3);
Qx_omega = w_w * eye(3);

% Temporarily leave R block zeros 
Qx_Rblock = zeros(9,9);




Qtilde = zeros(Nz, Nz);


% Build linear mapping L (3 x 9) such that theta ≈ L * R(:) (small-angle)
% ordering of R(:) is column-major: [R11; R21; R31; R12; R22; R32; R13; R23; R33]
L = zeros(3,9);
% theta1 = 0.5*(R32 - R23)
% indices: R(3,2) is element (row3,col2) -> column-major index: (col-1)*3 + row = (2-1)*3 +3 = 6
% R(2,3) -> (3-1)*3 +2 = 8
L(1,6) = 0.5;   % R32
L(1,8) = -0.5;  % -R23

% theta2 = 0.5*(R13 - R31)
% R13 index: (3-1)*3 + 1 = 7
% R31 index: (1-1)*3 + 3 = 3
L(2,7) = 0.5;   % R13
L(2,3) = -0.5;  % -R31

% theta3 = 0.5*(R21 - R12)
% R21 index: (2-1)*3 +1 = 2
% R12 index: (1-1)*3 +2 = 4
L(3,2) = 0.5;   % R21
L(3,4) = -0.5;  % -R12

% If you want theta relative to I: theta ≈ L*(R(:) - I(:)). However constant shift from I does not enter theta linear term because terms from identity cancel (I is symmetric).
% Build Q_theta (3x3)
Q_theta = w_theta * eye(3);

% Now convert to a 9x9 penalty on R(:): Q_R = L' * Q_theta * L
Q_R = L' * Q_theta * L;
Qx = blkdiag(Qx_pos, Qx_vel, Qx_Rblock, Qx_omega);

% Place into Qx
Qx(7:15, 7:15) = Q_R;
% Now Qx has position, velocity, R penalized via theta-approx, omega...
Qtilde(1:n,1:n) = Qx;
Qbar = kron(eye(Nh), Qtilde);

w_ft = 0.001;
w_M  = 10;
Ru = diag([w_ft, w_M, w_M, w_M]);


Rbar = kron(eye(Nh), Ru);

% constraints
ft_min = 10 ;
ft_max = 80;
M_max = 2;
M_min = -2;

U_lb = repmat([ft_min; M_min; M_min; M_min], Nh, 1);
U_ub = repmat([ft_max; M_max; M_max; M_max], Nh, 1);

%% Generate a random reference trajectory (higher-rate internal dt -> sample to Ts)
ref_dt = dt;                       % 0.001 as described
ref_steps_per_control = steps;

rng(1); % reproducible
mu_ref = [2;2;2;2];
Sigma_ref = diag([6,6,6,6]);


Nref_total = round(t_sim / ref_dt);
u_ref_high = mvnrnd(mu_ref, Sigma_ref, Nref_total)';   % 4 x Nref_total

for ii=1:Nref_total

    u_ref_high(1,ii) = u_ref_high(1,ii)+params.m*params.g;
end

% simulate reference nonlinear system (so we get reference states)
x_ref = zeros(n, Nref_total);
% initial ref state
p0 = zeros(3,1); v0=zeros(3,1); R0=eye(3); w0=zeros(3,1);
x = [p0; v0; R0(:); w0];
for k=1:Nref_total
    ucur = u_ref_high(:,k);
    for j=1:1 % integrate one dt (we treat each step as dt)
        % xdot = quad_dynamics(0, x, ucur, params);
        % x = x + ref_dt * xdot;
        x = rk4_step(x, ucur, params, dt);

        % reprojection
        Rm = reshape(x(7:15),3,3);
        [U_,~,V_] = svd(Rm);
        D = eye(3);
        if det(U_*V_') < 0, D(3,3) = -1; end
        Rm = U_ * D * V_';
        x(7:15) = Rm(:);
    end
    x_ref(:,k) = x;
end

ref_idx = 1:ref_steps_per_control:Nref_total;
Nref_ctrl = length(ref_idx);  % 120
x_ref_ctrl = x_ref(:, ref_idx);    % n x Nref_ctrl

% Extract reference signals at control rate (for plotting)
p_ref_ctrl      = x_ref_ctrl(1:3, :);
v_ref_ctrl      = x_ref_ctrl(4:6, :);

% orientation
theta_ref_ctrl  = zeros(3, size(x_ref_ctrl,2));
for k = 1:size(x_ref_ctrl,2)
    R_k = reshape(x_ref_ctrl(7:15,k),3,3);
    Theta_k = so3_log(R_k);
    theta_ref_ctrl(:,k) = vee(Theta_k);
end

omega_ref_ctrl  = x_ref_ctrl(16:18, :);



% form lifted z_ref_ctrl
z_ref_ctrl = zeros(Nz, Nref_ctrl);
for k=1:Nref_ctrl
    z_ref_ctrl(:,k) = observable_phi(x_ref_ctrl(:,k), p_obs);
end

%% Pre-allocate simulation storage
tvec = (0:Nsim-1) * Ts;
p_traj = zeros(3, Nsim);
v_traj = zeros(3, Nsim);
theta_traj = zeros(3, Nsim);
omega_traj = zeros(3, Nsim);
u_traj = zeros(nu, Nsim);

% initial plant state (same start as EDMD experiments)
x_true = [zeros(3,1); zeros(3,1); reshape(eye(3),9,1); zeros(3,1)];

% initial lifted state
z = observable_phi(x_true, p_obs);

% qp options
opts = optimoptions('quadprog','Display','none','Algorithm','interior-point-convex');




%% MPC LOOP (receding horizon)
for k = 1:Nsim
    % current time index in reference downsample (choose nearest)
    ref_idx_cur = min(k, Nref_ctrl);  % if sim longer than reference, use last point
    % Build stacked reference Y for horizon (use lifted z_ref_ctrl forward)
    Y = zeros(Nz*Nh,1);
    for i = 1:Nh
        idx = min(ref_idx_cur + (i-1), Nref_ctrl);
        Y((i-1)*Nz+1:i*Nz) = z_ref_ctrl(:, idx);
    end

    % Predictive cost terms 

    H = 2 * (Bqp' * Qbar * Bqp + Rbar);
    f = 2 * (Bqp' * Qbar * (Aqp * z - Y));  
    eps_reg = 1e-6;
    H = H + eps_reg * eye(size(H));

    H=0.5*(H+H');


    try
        [Uopt, ~, exitflag] = quadprog(H, f, [], [], [], [], U_lb, U_ub, [], opts);
        if exitflag <= 0

            Uopt = - (H + 1e-6*eye(size(H))) \ f;

        end
    catch
        Uopt = - pinv(H) * f;
    end

    % apply first control (u0)
    u0 = Uopt(1:nu);

    % store input
    u_traj(:, k) = u0;

    % propagate true nonlinear dynamics for one control interval Ts using internal dt steps
    x = x_true;
    for j = 1:steps
        % xdot = quad_dynamics(0, x, u0, params);
        % x = x + dt * xdot;
        x = rk4_step(x, u0, params, dt);

        Rm = reshape(x(7:15), 3, 3);
        [U_,~,V_] = svd(Rm);
        D = eye(3);
        if det(U_*V_') < 0, D(3,3) = -1; end
        Rm = U_ * D * V_';
        x(7:15) = Rm(:);
    end
    x_true = x;

    % update lifted for next iteration (Koopman prediction would also do z = A*z + B*u0)
    z = observable_phi(x_true, p_obs);

    % store states for plotting
    p_traj(:,k) = x_true(1:3);
    v_traj(:,k) = x_true(4:6);
    Rm = reshape(x_true(7:15),3,3);
    Theta = so3_log(Rm);
    theta_traj(:,k) = vee(Theta);
    omega_traj(:,k) = x_true(16:18);



end


%% --- PLOTTING: 6x2 grid with reference overlay ---

figure('Name','States (MPC vs Reference)','Color',[1 1 1],'Position',[100 100 1000 900]);

% Position
subplot(6,2,1); plot(tvec, p_traj(1,:), 'b', tvec, p_ref_ctrl(1,:), 'r--'); 
ylabel('p_x'); legend('actual','ref'); grid on;
subplot(6,2,2); plot(tvec, p_traj(2,:), 'b', tvec, p_ref_ctrl(2,:), 'r--'); 
ylabel('p_y'); legend('actual','ref'); grid on;
subplot(6,2,3); plot(tvec, p_traj(3,:), 'b', tvec, p_ref_ctrl(3,:), 'r--'); 
ylabel('p_z'); legend('actual','ref'); grid on;

% Velocity
subplot(6,2,4); plot(tvec, v_traj(1,:), 'b', tvec, v_ref_ctrl(1,:), 'r--'); 
ylabel('v_x'); legend('actual','ref'); grid on;
subplot(6,2,5); plot(tvec, v_traj(2,:), 'b', tvec, v_ref_ctrl(2,:), 'r--'); 
ylabel('v_y'); legend('actual','ref'); grid on;
subplot(6,2,6); plot(tvec, v_traj(3,:), 'b', tvec, v_ref_ctrl(3,:), 'r--'); 
ylabel('v_z'); legend('actual','ref'); grid on;

% Theta (orientation log-map)
subplot(6,2,7); plot(tvec, theta_traj(1,:), 'b', tvec, theta_ref_ctrl(1,:), 'r--'); 
ylabel('\theta_x'); legend('actual','ref'); grid on;
subplot(6,2,8); plot(tvec, theta_traj(2,:), 'b', tvec, theta_ref_ctrl(2,:), 'r--'); 
ylabel('\theta_y'); legend('actual','ref'); grid on;
subplot(6,2,9); plot(tvec, theta_traj(3,:), 'b', tvec, theta_ref_ctrl(3,:), 'r--'); 
ylabel('\theta_z'); legend('actual','ref'); grid on;

% Angular velocity
subplot(6,2,10); plot(tvec, omega_traj(1,:), 'b', tvec, omega_ref_ctrl(1,:), 'r--'); 
ylabel('\omega_x'); legend('actual','ref'); grid on;
subplot(6,2,11); plot(tvec, omega_traj(2,:), 'b', tvec, omega_ref_ctrl(2,:), 'r--'); 
ylabel('\omega_y'); legend('actual','ref'); grid on;
subplot(6,2,12); plot(tvec, omega_traj(3,:), 'b', tvec, omega_ref_ctrl(3,:), 'r--'); 
ylabel('\omega_z'); legend('actual','ref'); grid on;

sgtitle('Koopman-MPC Tracking: Actual vs Reference States');

figure('Name','Inputs (2x2)','Color',[1 1 1],'Position',[200 200 700 500]);
subplot(2,2,1); plot(tvec, u_traj(1,:)); ylabel('f_t'); grid on;
subplot(2,2,2); plot(tvec, u_traj(2,:)); ylabel('M_1'); grid on;
subplot(2,2,3); plot(tvec, u_traj(3,:)); ylabel('M_2'); grid on;
subplot(2,2,4); plot(tvec, u_traj(4,:)); ylabel('M_3'); grid on;
sgtitle('Koopman-MPC Inputs (Thrust and Moments)');

disp('Simulation complete.');