clc; clear all; close all;

%% Parameters
params.m = 4.34;
params.J = diag([0.0820; 0.0845; 0.1377]);
params.g = 9.81;

ts    = 0.01;     % EDMD sample period (s)
dt    = 0.001;    % integration step (s)
steps = round(ts/dt);
T     = 0.1;      % length of each training trajectory (s)
Ntraj = 100;

%% Generate data for EDMD (training)
X_list  = {};
Xp_list = {};
U_list  = {};

for tr = 1:Ntraj
    % initial condition
    p0    = zeros(3,1);
    v0    = zeros(3,1);
    R0    = eye(3);
    omega0 = zeros(3,1);
    x     = [p0; v0; R0(:); omega0];   % 18x1

    mu_u    = [0;0;0;0];
    Sigma_u = diag([2;2;2;2]);
    u       = mvnrnd(mu_u, Sigma_u).';
    u(1)    = u(1) + params.m*params.g;  % add mg to thrust

    Nsteps = round(T/ts);
    for k = 1:Nsteps
        x_k = x;
        % integrate true dynamics over one EDMD sample period
        for j = 1:steps
            x = rk4_step(x, u, params, dt);
        end
        x_kp1 = x;

        X_list{end+1}  = x_k;
        Xp_list{end+1} = x_kp1;
        U_list{end+1}  = u;
    end
end

% convert to matrices
X   = cell2mat(X_list);      % n x K
Xp  = cell2mat(Xp_list);     % n x K
U   = cell2mat(U_list);      % 4 x K

%% Build lifted data Z,Zp
p   = 3;                     % observable order
K   = size(X,2);
n   = size(X,1);             % n = 18
Nz  = n + 9*p;               % 18 + 27 = 45

Z  = zeros(Nz, K);
Zp = zeros(Nz, K);
for k = 1:K
    Z(:, k)  = observable_phi(X(:,  k), p);
    Zp(:, k) = observable_phi(Xp(:, k), p);
end

%% EDMD with control
ZU     = [Z; U];  
lambda = 1e-6;                       % regularization
G      = Zp * ZU' / (ZU*ZU' + lambda*eye(size(ZU,1)));

A  = G(:, 1:Nz);
B  = G(:, Nz+1:end);
C  = [eye(n), zeros(n, 9*p)];        % n x Nz

%% Validation: generate 50 new trajectories
Nval_traj = 50;
Tval      = 0.1;
Nsteps    = round(Tval/ts);

mu_val    = [0;0;0;0];
Sigma_val = diag([4;4;4;4]);

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

% logs for full state trajectories (for plotting)
X_true_log = [];
X_pred_log = [];
U_val_log=[];

for tr = 1:Nval_traj
    % initial true state
    p0 = zeros(3,1);
    v0 = zeros(3,1);
    R0 = eye(3);
    w0 = zeros(3,1);
    x_true = [p0; v0; R0(:); w0];

    % corresponding lifted state
    z = observable_phi(x_true, p);

    % random constant input for this trajectory
    u     = mvnrnd(mu_val, Sigma_val).';
    u(1)  = u(1) + params.m*params.g;

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

        % store for plotting
        X_true_log = [X_true_log, x_true];
        X_pred_log = [X_pred_log, x_pred];
        U_val_log=[U_val_log,u];

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

        % --- theta via same routine as rmse: RotToRPY_ZXY(QuatToRot(.)) ---
        % true
        q_true  = RotToQuat(R_true);            
        bRw_t   = QuatToRot(q_true);             % rotation matrix (body from world)
        [rt, pt, yt] = RotToRPY_ZXY(bRw_t);      % roll, pitch, yaw
        theta_true = [rt; pt; yt];

        % predicted
        q_hat   = RotToQuat(R_hat);
        bRw_h   = QuatToRot(q_hat);
        [rh, ph, yh] = RotToRPY_ZXY(bRw_h);
        theta_hat = [rh; ph; yh];

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

%% RMSE and nRMSE
RMSE_p     = sqrt(mean(err_p.^2));
RMSE_v     = sqrt(mean(err_v.^2));
RMSE_R     = sqrt(mean(err_R.^2));
RMSE_w     = sqrt(mean(err_w.^2));
RMSE_theta = sqrt(mean(err_th.^2));

eps_den      = 1e-8;
nRMSE_p      = 100*sqrt(num_p  / max(den_p, eps_den));
nRMSE_v      = 100*sqrt(num_v  / max(den_v, eps_den));
nRMSE_R      = 100*sqrt(num_R  / max(den_R, eps_den));
nRMSE_w      = 100*sqrt(num_w  / max(den_w, eps_den));
nRMSE_theta  = 100*sqrt(num_th / max(den_th, eps_den));

fprintf("position RMSE = %.4f%%\n", nRMSE_p);
fprintf("velocity RMSE = %.4f%%\n", nRMSE_v);
fprintf("theta RMSE = %.4f%%\n",   nRMSE_theta);
fprintf("omega RMSE = %.4f%%\n",  nRMSE_w);

%% Build 12‑state plots: px,py,pz, vx,vy,vz, thetax,thetay,thetaz, wx,wy,wz

Ntot = size(X_true_log,2);
tvec = (0:Ntot-1)*ts;

% unpack true and predicted
p_true_all = X_true_log(1:3,:);
v_true_all = X_true_log(4:6,:);
R_true_all = X_true_log(7:15,:);
w_true_all = X_true_log(16:18,:);

p_hat_all  = X_pred_log(1:3,:);
v_hat_all  = X_pred_log(4:6,:);
R_hat_all  = X_pred_log(7:15,:);
w_hat_all  = X_pred_log(16:18,:);

% compute theta for all steps using same ZXY routine
theta_true_all = zeros(3,Ntot);
theta_hat_all  = zeros(3,Ntot);
for k = 1:Ntot
    Rk_true = reshape(R_true_all(:,k),3,3);
    Rk_hat  = reshape(R_hat_all(:,k),3,3);

    q_t = RotToQuat(Rk_true);
    q_h = RotToQuat(Rk_hat);

    [rt, pt, yt] = RotToRPY_ZXY(QuatToRot(q_t));
    [rh, ph, yh] = RotToRPY_ZXY(QuatToRot(q_h));

    theta_true_all(:,k) = [rt; pt; yt];
    theta_hat_all(:,k)  = [rh; ph; yh];
end

% % Plot
% figure;
% % position
% subplot(6,2,1);
% plot(tvec, p_true_all(1,:), tvec, p_hat_all(1,:),'--','LineWidth',1.5);
% ylabel('p_x'); grid on; legend('true','EDMD');
% 
% subplot(6,2,3);
% plot(tvec, p_true_all(2,:), tvec, p_hat_all(2,:),'--','LineWidth',1.5);
% ylabel('p_y'); grid on;
% 
% subplot(6,2,5);
% plot(tvec, p_true_all(3,:), tvec, p_hat_all(3,:),'--','LineWidth',1.5);
% ylabel('p_z'); grid on;
% 
% % velocity
% subplot(6,2,7);
% plot(tvec, v_true_all(1,:), tvec, v_hat_all(1,:),'--','LineWidth',1.5);
% ylabel('v_x'); grid on;
% 
% subplot(6,2,9);
% plot(tvec, v_true_all(2,:), tvec, v_hat_all(2,:),'--','LineWidth',1.5);
% ylabel('v_y'); grid on;
% 
% subplot(6,2,11);
% plot(tvec, v_true_all(3,:), tvec, v_hat_all(3,:),'--','LineWidth',1.5);
% ylabel('v_z'); xlabel('time (s)'); grid on;
% 
% % orientation (roll, pitch, yaw)
% subplot(6,2,2);
% plot(tvec, theta_true_all(1,:), tvec, theta_hat_all(1,:),'--','LineWidth',1.5);
% ylabel('\theta_x'); grid on;
% 
% subplot(6,2,4);
% plot(tvec, theta_true_all(2,:), tvec, theta_hat_all(2,:),'--','LineWidth',1.5);
% ylabel('\theta_y'); grid on;
% 
% subplot(6,2,6);
% plot(tvec, theta_true_all(3,:), tvec, theta_hat_all(3,:),'--','LineWidth',1.5);
% ylabel('\theta_z'); grid on;
% 
% % angular velocity
% subplot(6,2,8);
% plot(tvec, w_true_all(1,:), tvec, w_hat_all(1,:),'--','LineWidth',1.5);
% ylabel('\omega_x'); grid on;
% 
% subplot(6,2,10);
% plot(tvec, w_true_all(2,:), tvec, w_hat_all(2,:),'--','LineWidth',1.5);
% ylabel('\omega_y'); grid on;
% 
% subplot(6,2,12);
% plot(tvec, w_true_all(3,:), tvec, w_hat_all(3,:),'--','LineWidth',1.5);
% ylabel('\omega_z'); xlabel('time (s)'); grid on;
% 
% sgtitle('EDMD model vs true dynamics: 12 states');
% 
% 
% 
% 
% %% Plot validation control inputs: ft, M1, M2, M3
% 
% t_u = (0:size(U_val_log,2)-1)*ts;    % same sampling as states
% 
% figure;
% subplot(4,1,1);
% plot(t_u, U_val_log(1,:),'LineWidth',1.5);
% ylabel('f_t'); grid on;
% title('Validation control inputs');
% 
% subplot(4,1,2);
% plot(t_u, U_val_log(2,:),'LineWidth',1.5);
% ylabel('M_1'); grid on;
% 
% subplot(4,1,3);
% plot(t_u, U_val_log(3,:),'LineWidth',1.5);
% ylabel('M_2'); grid on;
% 
% subplot(4,1,4);
% plot(t_u, U_val_log(4,:),'LineWidth',1.5);
% ylabel('M_3'); xlabel('time (s)'); grid on;


%% === MPC on lifted state (Koopman-MPC) ===

Nh   = 10;          % prediction horizon
Ts   = 0.01;        % control period (same as your EDMD sample time)
t_sim = 1.2;
Nsim = round(t_sim / Ts);

Nz = size(A,1);     % lifted dimension
n  = size(C,1);     % original physical state dimension (18 here)
nu = size(B,2);     % number of inputs (4)
% Block prediction model in lifted space: z_{k+i} = A^i z_k + ...
A_hat = zeros(Nz*Nh, Nz);
B_hat = zeros(Nz*Nh, nu*Nh);

A_power = eye(Nz);
for i = 1:Nh
    A_power = A_power * A;                  % A^i
    A_hat((i-1)*Nz+1:i*Nz, :) = A_power;
    for j = 1:i
        Aij = A^(i-j) * B;                  % contribution of u_{k+j-1} to z_{k+i}
        B_hat((i-1)*Nz+1:i*Nz, (j-1)*nu+1:j*nu) = Aij;
    end
end


% Weights for physical 12-state [p; v; theta; omega]
Qp = 1e6 * eye(3);      % position
Qv = 1e5* eye(3);      % velocity
% Qa=[10,0,0;0,10,0;0,0,100];
Qa = 1e6 * eye(3);      % angles (roll,pitch,yaw)
Qw = 1e4 * eye(3);      % angular rates

% Per-step lifted-state weight Q_i (Nz x Nz):
% only the first 12 physical states are penalized; the rest of z is unweighted.
Q_i = zeros(Nz, Nz);
Q_i(1:12,1:12) = blkdiag(Qp, Qv, Qa, Qw);

P = Q_i;                % terminal weight (can tune separately)

% Input weights
R_i = diag([1e2; 1e1; 1e1; 1e1]);   % [f_t, M1, M2, M3]

% Stack over horizon
Q_hat = [];
R_hat = [];
for i = 1:Nh
    Q_hat = blkdiag(Q_hat, Q_i);
    R_hat = blkdiag(R_hat, R_i);
end
Q_hat(end-Nz+1:end, end-Nz+1:end) = P;   % terminal cost on last z


% Simple box constraints on inputs
ft_min = 10;
ft_max = 80;
M_max  = 2;
M_min  = -2;

U_lb = repmat([ft_min; M_min; M_min; M_min], Nh, 1);
U_ub = repmat([ft_max; M_max; M_max; M_max], Nh, 1);

% Inequality matrices for quadprog: A_ineq * U <= b_ineq
% Here just bound constraints, so use them as lower/upper vectors.
A_ineq = [];
b_ineq = [];


%% Reference trajectory generation at high rate dt, then downsample to Ts
ref_dt = dt;                 % internal integration step
steps_per_ctrl = round(Ts/ref_dt);
Nref_total = round(t_sim / ref_dt);

rng(1);
mu_ref    = [2;1;1;1];
Sigma_ref = diag([1,1,1,1]);

u_ref_high = mvnrnd(mu_ref, Sigma_ref, Nref_total)';  % 4 x Nref_total
for ii = 1:Nref_total
    u_ref_high(1,ii) = u_ref_high(1,ii) + params.m*params.g;
end

x_ref = zeros(n, Nref_total);
x = [zeros(3,1); zeros(3,1); reshape(eye(3),9,1); zeros(3,1)];
for k = 1:Nref_total
    ucur = u_ref_high(:,k);
    x = rk4_step(x, ucur, params, ref_dt);
    % reproject R
    % Rm = reshape(x(7:15),3,3);
    % [U_,~,V_] = svd(Rm); D = eye(3);
    % if det(U_*V_') < 0, D(3,3) = -1; end
    % Rm = U_*D*V_';
    % x(7:15) = Rm(:);
    x_ref(:,k) = x;
end

% Downsample to control rate Ts
ref_idx     = 1:steps_per_ctrl:Nref_total;
Nref_ctrl   = length(ref_idx);
x_ref_ctrl  = x_ref(:, ref_idx);      % n x Nref_ctrl

% Lifted reference z_ref at control rate
z_ref_ctrl = zeros(Nz, Nref_ctrl);
for k = 1:Nref_ctrl
    z_ref_ctrl(:,k) = observable_phi(x_ref_ctrl(:,k), p);   % p is your observable order
end


%% Storage for plotting
tvec        = (0:Nsim-1)*Ts;
x_traj      = zeros(n, Nsim);      % full true state
u_traj      = zeros(nu, Nsim);     % inputs

% Initial state and lifted state
x_true = [zeros(3,1); zeros(3,1); reshape(eye(3),9,1); zeros(3,1)];
z      = observable_phi(x_true, p);

opts = optimoptions('quadprog','Display','none','Algorithm','interior-point-convex');

for k = 1:Nsim
    % Current reference index
    ref_idx_cur = min(k, Nref_ctrl);

    % Build stacked reference Y = [z_ref(k+1); ...; z_ref(k+Nh)]
    Y = zeros(Nz*Nh,1);
    for i = 1:Nh
        idx = min(ref_idx_cur + (i-1), Nref_ctrl);
        Y((i-1)*Nz+1:i*Nz) = z_ref_ctrl(:, idx);
    end

    % Quadratic cost: 1/2 U' G U + U' F
    G = 2*(R_hat + B_hat' * Q_hat * B_hat);
    F = 2*B_hat' * Q_hat * (A_hat * z - Y);

    % small regularization, symmetrize
    eps_reg = 1e-6;
    G = G + eps_reg * eye(size(G));
    G = 0.5*(G + G');

    % Solve QP for optimal input sequence U = [u_0; ...; u_{Nh-1}]
    try
        [Uopt,~,exitflag] = quadprog(G, F, A_ineq, b_ineq, [], [], U_lb, U_ub, [], opts);
        if exitflag <= 0
            Uopt = -(G + 1e-6*eye(size(G)))\F;
        end
    catch
        Uopt = -pinv(G)*F;
    end

    u0 = Uopt(1:nu);
    u_traj(:,k) = u0;

    % Propagate true nonlinear plant over Ts using dt
    x = x_true;
    for j = 1:steps
        x = rk4_step(x, u0, params, dt);
        % Rm = reshape(x(7:15),3,3);
        % [U_,~,V_] = svd(Rm); D = eye(3);
        % if det(U_*V_') < 0, D(3,3) = -1; end
        % Rm = U_*D*V_';
        % x(7:15) = Rm(:);
    end
    x_true = x;
    x_traj(:,k) = x_true;

    % Update lifted state from measurement (closed-loop Koopman MPC)
    z = observable_phi(x_true, p);
end


% Extract physical trajectories and reference at control rate
p_traj     = x_traj(1:3,:);
v_traj     = x_traj(4:6,:);
R_traj_all = x_traj(7:15,:);
w_traj     = x_traj(16:18,:);

p_ref_ctrl   = x_ref_ctrl(1:3,1:Nsim);      % align lengths
v_ref_ctrl   = x_ref_ctrl(4:6,1:Nsim);
R_ref_all    = x_ref_ctrl(7:15,1:Nsim);
w_ref_ctrl   = x_ref_ctrl(16:18,1:Nsim);

theta_traj     = zeros(3,Nsim);
theta_ref_ctrl = zeros(3,Nsim);
for k = 1:Nsim
    Rk    = reshape(R_traj_all(:,k),3,3);
    Rkref = reshape(R_ref_all(:,k),3,3);
    [r,pit,yw]     = RotToRPY_ZXY(Rk);      % your function
    [rr,pp,yy]     = RotToRPY_ZXY(Rkref);
    theta_traj(:,k)     = [r;pit;yw];
    theta_ref_ctrl(:,k) = [rr;pp;yy];
end

% Trim reference to simulation length
x_ref_plot = x_ref_ctrl(:,1:Nsim);

% True closed-loop trajectory
x_mpc_plot = x_traj(:,1:Nsim);

t_ctrl = (0:Nsim-1)*Ts;

% Parse states
p_ref   = x_ref_plot(1:3,:);
v_ref   = x_ref_plot(4:6,:);
R_ref   = x_ref_plot(7:15,:);
w_ref   = x_ref_plot(16:18,:);

p_mpc   = x_mpc_plot(1:3,:);
v_mpc   = x_mpc_plot(4:6,:);
R_mpc   = x_mpc_plot(7:15,:);
w_mpc   = x_mpc_plot(16:18,:);

theta_ref = zeros(3,Nsim);
theta_mpc = zeros(3,Nsim);
for k = 1:Nsim
    Rr = reshape(R_ref(:,k),3,3);
    Rm = reshape(R_mpc(:,k),3,3);

    [rr,pr,yr] = RotToRPY_ZXY(Rr);
    [rm,pm,ym] = RotToRPY_ZXY(Rm);

    theta_ref(:,k) = [rr;pr;yr];
    theta_mpc(:,k) = [rm;pm;ym];
end

figure;

% position
subplot(6,2,1);
plot(t_ctrl, p_ref(1,:), t_ctrl, p_mpc(1,:),'--','LineWidth',1.5);
ylabel('p_x'); grid on; legend('ref','MPC');

subplot(6,2,3);
plot(t_ctrl, p_ref(2,:), t_ctrl, p_mpc(2,:),'--','LineWidth',1.5);
ylabel('p_y'); grid on;

subplot(6,2,5);
plot(t_ctrl, p_ref(3,:), t_ctrl, p_mpc(3,:),'--','LineWidth',1.5);
ylabel('p_z'); grid on;

% velocity
subplot(6,2,7);
plot(t_ctrl, v_ref(1,:), t_ctrl, v_mpc(1,:),'--','LineWidth',1.5);
ylabel('v_x'); grid on;

subplot(6,2,9);
plot(t_ctrl, v_ref(2,:), t_ctrl, v_mpc(2,:),'--','LineWidth',1.5);
ylabel('v_y'); grid on;

subplot(6,2,11);
plot(t_ctrl, v_ref(3,:), t_ctrl, v_mpc(3,:),'--','LineWidth',1.5);
ylabel('v_z'); xlabel('time (s)'); grid on;

% orientation (roll, pitch, yaw)
subplot(6,2,2);
plot(t_ctrl, theta_ref(1,:), t_ctrl, theta_mpc(1,:),'--','LineWidth',1.5);
ylabel('\theta_x'); grid on;

subplot(6,2,4);
plot(t_ctrl, theta_ref(2,:), t_ctrl, theta_mpc(2,:),'--','LineWidth',1.5);
ylabel('\theta_y'); grid on;

subplot(6,2,6);
plot(t_ctrl, theta_ref(3,:), t_ctrl, theta_mpc(3,:),'--','LineWidth',1.5);
ylabel('\theta_z'); grid on;

% angular velocity
subplot(6,2,8);
plot(t_ctrl, w_ref(1,:), t_ctrl, w_mpc(1,:),'--','LineWidth',1.5);
ylabel('\omega_x'); grid on;

subplot(6,2,10);
plot(t_ctrl, w_ref(2,:), t_ctrl, w_mpc(2,:),'--','LineWidth',1.5);
ylabel('\omega_y'); grid on;

subplot(6,2,12);
plot(t_ctrl, w_ref(3,:), t_ctrl, w_mpc(3,:),'--','LineWidth',1.5);
ylabel('\omega_z'); xlabel('time (s)'); grid on;

sgtitle('Reference vs MPC trajectory (12 states)');


% Random reference controls at control rate
u_ref_ctrl = u_ref_high(:, ref_idx);      % 4 x Nref_ctrl
u_ref_plot = u_ref_ctrl(:,1:Nsim);        % match Nsim

% MPC controls already stored as u_traj (4 x Nsim)
u_mpc_plot = u_traj(:,1:Nsim);

figure;
subplot(4,1,1);
plot(t_ctrl, u_ref_plot(1,:), t_ctrl, u_mpc_plot(1,:),'--','LineWidth',1.5);
ylabel('f_t'); grid on; legend('ref-rand','MPC');
title('Control: reference random vs MPC');

subplot(4,1,2);
plot(t_ctrl, u_ref_plot(2,:), t_ctrl, u_mpc_plot(2,:),'--','LineWidth',1.5);
ylabel('M_1'); grid on;

subplot(4,1,3);
plot(t_ctrl, u_ref_plot(3,:), t_ctrl, u_mpc_plot(3,:),'--','LineWidth',1.5);
ylabel('M_2'); grid on;

subplot(4,1,4);
plot(t_ctrl, u_ref_plot(4,:), t_ctrl, u_mpc_plot(4,:),'--','LineWidth',1.5);
ylabel('M_3'); xlabel('time (s)'); grid on;
