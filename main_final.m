

clc; clear; close all;

%% Parameters
params.m = 4.34;
params.J = diag([0.0820;0.0845;0.1377]);
params.g = 9.81;
% J1 = 0.02;
% J2 = 0.02;
% J3 = 0.04;
% params.J = diag([J1, J2, J3]);
% params.m = 2;
% integration and sampling
dt = 0.001;      % internal integration step
Ts = 0.01;       % sampling / control update (100 Hz)
steps = round(Ts/dt);
params.d = 0.169;
% params.ctf = 0.0135;
% training dataset
T_traj = 0.1;    % trajectory length (s)
Ntraj = 100;     % number of training trajectories

%% --- Collect snapshots (training) ---
X_list = {};
Xp_list = {};
U_list = {};

rng(0);
for tr = 1:Ntraj
    % initial condition
    p0 = zeros(3,1);
    v0 = zeros(3,1);
    R0 = eye(3);
    w0 = zeros(3,1);
    x = [p0; v0; R0(:); w0];

    % random constant input per trajectory (as in original script)
    mu_u = [0;0;0;0];
    Sigma_u = diag([10;10;10;10]);
    u = mvnrnd(mu_u, Sigma_u).';
    % enforce liftoff thrust in training (same as your earlier script)
    % u(1) = max(u(1), 1.2*params.m*params.g);
    u(1) = u(1)+params.m*params.g;

    Nsteps = round(T_traj / Ts);
    for k = 1:Nsteps
        x_k = x;
        % propagate for Ts
        for j = 1:steps
            xdot = quad_dynamics(0, x, u, params);
            % x = x + dt * xdot;
            x = rk4_step(x, u, params, dt);
            % reproject R to SO(3) via polar with det correction
            Rm = reshape(x(7:15), 3, 3);
            [U_,~,V_] = svd(Rm);
            D = eye(3);
            if det(U_*V_') < 0, D(3,3) = -1; end
            Rm = U_ * D * V_';
            x(7:15) = Rm(:);
        end
        x_kp1 = x;
        X_list{end+1} = x_k;
        Xp_list{end+1} = x_kp1;
        U_list{end+1} = u;
    end
end

% convert to matrices
X  = cell2mat(X_list);    % n x K
Xp = cell2mat(Xp_list);
U  = cell2mat(U_list);

% lifting parameter
p_obs = 3;       % number of pow-hat observables per axis (paper uses p=3)
n = size(X,1);   % original state dim (should be 18)
Nz = n + 9*p_obs; % lifted dimension (should be 45)

%% --- Build lifted data Z, Zp ---
K = size(X,2);
Z = zeros(Nz, K);
Zp = zeros(Nz, K);
for k = 1:K
    Z(:,k)  = observable_phi(X(:,k), p_obs);
    Zp(:,k) = observable_phi(Xp(:,k), p_obs);
end

%% --- EDMD identification (robust pseudo-inverse) ---
ZU = [Z; U];    % (Nz+nu) x K
G = Zp * pinv(ZU);    % Nz x (Nz+nu)
A = G(:, 1:Nz);
B = G(:, Nz+1:end);

% output mapping back to original states
C = [eye(n), zeros(n, Nz-n)];   % n x Nz

%% --- VALIDATION (50 trajectories) compute RMSEs ---
Nval_traj = 50;
Tval = 0.1;
Nsteps_val = round(Tval / Ts);

% validation sampling (match training distribution or as intended)
mu_val = [0;0;0;0];
Sigma_val = diag([20;20;20;20]);

% error accumulators
err_p = []; err_v = []; err_R = []; err_w = []; err_th = [];
num_p = 0; den_p = 0;
num_v = 0; den_v = 0;
num_R = 0; den_R = 0;
num_w = 0; den_w = 0;
num_th = 0; den_th = 0;

for tr = 1:Nval_traj
    x_true = [zeros(3,1); zeros(3,1); reshape(eye(3),9,1); zeros(3,1)];
    z = observable_phi(x_true, p_obs);

    u = mvnrnd(mu_val, Sigma_val).';
    % u(1) = max(u(1), 1.2*params.m*params.g);   % match training liftoff rule
    u(1) = u(1)+params.m*params.g;

    for k = 1:Nsteps_val
        x = x_true;
        for j = 1:steps
            xdot = quad_dynamics(0, x, u, params);
            % x = x + dt * xdot;
            x = rk4_step(x, u, params, dt);
            Rm = reshape(x(7:15),3,3);
            [U_,~,V_] = svd(Rm);
            D = eye(3);
            if det(U_*V_') < 0, D(3,3) = -1; end
            Rm = U_ * D * V_';
            x(7:15) = Rm(:);
        end
        x_true = x;

        % Koopman prediction one step
        z = A*z + B*u;
        x_pred = C*z;

        % unpack and errors
        p_true = x_true(1:3); v_true = x_true(4:6); R_true = reshape(x_true(7:15),3,3); w_true = x_true(16:18);
        p_hat  = x_pred(1:3); v_hat  = x_pred(4:6); R_hat  = reshape(x_pred(7:15),3,3); w_hat  = x_pred(16:18);

        Theta_true = so3_log(R_true); Theta_hat = so3_log(R_hat);
        theta_true = vee(Theta_true); theta_hat = vee(Theta_hat);

        e_p  = norm(p_hat - p_true);
        e_v  = norm(v_hat - v_true);
        e_R  = norm(R_hat - R_true, 'fro');
        e_w  = norm(w_hat - w_true);
        e_th = norm(theta_hat - theta_true);

        err_p(end+1,1) = e_p; err_v(end+1,1) = e_v;
        err_R(end+1,1) = e_R; err_w(end+1,1) = e_w;
        err_th(end+1,1) = e_th;

        num_p = num_p + e_p^2; den_p = den_p + norm(p_true)^2;
        num_v = num_v + e_v^2; den_v = den_v + norm(v_true)^2;
        num_R = num_R + e_R^2; den_R = den_R + norm(R_true,'fro')^2;
        num_w = num_w + e_w^2; den_w = den_w + norm(w_true)^2;
        num_th= num_th + e_th^2; den_th = den_th + norm(theta_true)^2;
    end
end

RMSE_p     = sqrt(mean(err_p.^2));
RMSE_v     = sqrt(mean(err_v.^2));
RMSE_R     = sqrt(mean(err_R.^2));
RMSE_w     = sqrt(mean(err_w.^2));
RMSE_theta = sqrt(mean(err_th.^2));

eps_den = 1e-8;
nRMSE_p     = 100 * sqrt(num_p / max(den_p, eps_den));
nRMSE_v     = 100 * sqrt(num_v / max(den_v, eps_den));
nRMSE_R     = 100 * sqrt(num_R / max(den_R, eps_den));
nRMSE_w     = 100 * sqrt(num_w / max(den_w, eps_den));
nRMSE_theta = 100 * sqrt(num_th / max(den_th, eps_den));

fprintf("Validation normalized RMSE (percent):\n");
fprintf(" position: %.4f%%\n velocity: %.4f%%\n theta: %.4f%%\n omega: %.4f%%\n", ...
    nRMSE_p, nRMSE_v, nRMSE_theta, nRMSE_w);

%% --------------------------------------------
%% Koopman-MPC setup (lifted-space MPC as in paper)
%% --------------------------------------------

Nh = 10;            % prediction horizon
t_sim = 1.2;        % simulation time
Nsim = round(t_sim / Ts);
nu = size(B,2);    % number of control inputs
% Build stacked Aqp, Bqp
Aqp = zeros(Nz*Nh, Nz);
Bqp = zeros(Nz*Nh, nu*Nh);
A_power = eye(Nz);
for i = 1:Nh
    A_power = A_power * A;   % A^i
    Aqp((i-1)*Nz+1:i*Nz, :) = A_power;
    for j = 1:i
        Bqp((i-1)*Nz+1:i*Nz, (j-1)*nu+1:j*nu) = A^(i-j) * B;
    end
end

% Cost: penalize only original n states inside lifted vector
Qx = diag([1e6,1e6,1e6, 1e5,1e5,1e6, 10*ones(1,9), 1e5,1e5,1e5]); % n x n
% Build Qtilde (Nz x Nz) with Qx in top-left and zeros elsewhere
Qtilde = zeros(Nz, Nz);
Qtilde(1:n, 1:n) = Qx;
Qbar = kron(eye(Nh), Qtilde);

% Control regularization (block)
% Ru = 0.5 * eye(nu);
Ru = diag([1e1;1e0;1e0;1e0]);
Rbar = kron(eye(Nh), Ru);

% Input constraints
ft_min = 0.5 * params.m * params.g;
ft_max = 5   * params.m * params.g;
M_max = 20; M_min = -20;

% u_lb_step = [ft_min; M_min; M_min; M_min];
% u_ub_step = [ft_max; M_max; M_max; M_max];
u_lb_step =-10.*[0;0.05;0.05;0.05];
u_ub_step = 10.*[2;0.05;0.05;0.05];
U_lb = repmat(u_lb_step, Nh, 1);
U_ub = repmat(u_ub_step, Nh, 1);

%% Generate random reference trajectory (higher-rate dt=0.001, different distribution)
rng(1);
mu_ref = [2;2;2;2];
Sigma_ref = diag([30,30,30,30]);

ref_dt = dt;
Nref_total = round(t_sim / ref_dt);
u_ref_high = mvnrnd(mu_ref, Sigma_ref, Nref_total)';  % 4 x Nref_total
% avoid tiny thrust
for ii = 1:Nref_total
    % if u_ref_high(1,ii) < 0.2*params.m*params.g
    %     u_ref_high(1,ii) = 0.2*params.m*params.g;
    % end
    u_ref_high(1,ii) = u_ref_high(1,ii)+params.m*params.g;
end

% simulate reference nonlinear system
x_ref = zeros(n, Nref_total);
x = [zeros(3,1); zeros(3,1); reshape(eye(3),9,1); zeros(3,1)];
% for k = 1:Nref_total
%     ucur = u_ref_high(:,k);
%     xdot = quad_dynamics(0, x, ucur, params);
%     x = x + ref_dt * xdot;
%     Rm = reshape(x(7:15),3,3);
%     [U_,~,V_] = svd(Rm);
%     D = eye(3);
%     if det(U_*V_') < 0, D(3,3) = -1; end
%     Rm = U_*D*V_'; %#ok<*NASGU>
%     x(7:15) = Rm(:);
%     x_ref(:,k) = x;
% end
for k=1:Nref_total
    ucur = u_ref_high(:,k);
    for j=1:1 % integrate one dt (we treat each step as dt)
        xdot = quad_dynamics(0, x, ucur, params);
        % x = x + ref_dt * xdot;
        x = rk4_step(x, u, params, dt);
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

% downsample reference to control-rate Ts
ref_idx = 1:steps:Nref_total;
x_ref_ctrl = x_ref(:, ref_idx);    % n x Nref_ctrl
Nref_ctrl = size(x_ref_ctrl,2);

% reference in lifted coordinates (control rate)
z_ref_ctrl = zeros(Nz, Nref_ctrl);
for k = 1:Nref_ctrl
    z_ref_ctrl(:,k) = observable_phi(x_ref_ctrl(:,k), p_obs);
end

% reference signals for plotting (downsampled)
p_ref_ctrl = x_ref_ctrl(1:3, :);
v_ref_ctrl = x_ref_ctrl(4:6, :);
theta_ref_ctrl = zeros(3, Nref_ctrl);
for k = 1:Nref_ctrl
    Rk = reshape(x_ref_ctrl(7:15,k), 3,3);
    Theta = so3_log(Rk);
    theta_ref_ctrl(:,k) = vee(Theta);
end
omega_ref_ctrl = x_ref_ctrl(16:18, :);

%% Pre-allocate simulation
tvec = (0:Nsim-1) * Ts;
p_traj = zeros(3, Nsim); v_traj = zeros(3, Nsim);
theta_traj = zeros(3, Nsim); omega_traj = zeros(3, Nsim);
u_traj = zeros(nu, Nsim);

% initial real plant state
x_true = [zeros(3,1); zeros(3,1); reshape(eye(3),9,1); zeros(3,1)];
z = observable_phi(x_true, p_obs);

% qp options
opts = optimoptions('quadprog','Display','none','Algorithm','interior-point-convex');

%% MPC loop (receding horizon)
for k = 1:Nsim
    ref_idx_cur = min(k, Nref_ctrl);

    % form stacked Y = lifted reference trajectory over horizon
    Y = zeros(Nz*Nh, 1);
    for i = 1:Nh
        idx = min(ref_idx_cur + (i-1), Nref_ctrl);
        Y((i-1)*Nz+1:i*Nz) = z_ref_ctrl(:, idx);
    end

    % Build QP via helper build_mpc
    [H, G, Aineq, bineq, lb, ub] = build_mpc(Aqp, Bqp, Qtilde, Rbar, z, Y, u_lb_step, u_ub_step, Nh);

    % ensure H symmetric
    H = 0.5*(H + H');

    % solve QP: minimize 0.5 U' H U + U' G   subject to Aineq * U <= bineq
    try
        [Uopt, ~, exitflag] = quadprog(H, G, Aineq, bineq, [], [], [], [], [], opts);
        if exitflag ~= 1
            Uopt = - H \ G;  % fallback unconstrained
        end
    catch
        Uopt = - H \ G;
    end

    u0 = Uopt(1:nu);
    u_traj(:,k) = u0;

    % propagate true nonlinear dynamics for one Ts with internal dt
    x = x_true;
    for j = 1:steps
        xdot = quad_dynamics(0, x, u0, params);
        x = x + dt * xdot;
        Rm = reshape(x(7:15),3,3);
        [U_,~,V_] = svd(Rm);
        D = eye(3);
        if det(U_*V_') < 0, D(3,3) = -1; end
        Rm = U_ * D * V_';
        x(7:15) = Rm(:);
    end
    x_true = x;

    % update lift from true state to avoid drift
    z = observable_phi(x_true, p_obs);

    % store states
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




% %% PLOTTING: states (6x2) and inputs (2x2) with reference overlay
% % Trim reference arrays to simulation length Nsim
% idx_ref_plot = 1:min(Nsim, size(p_ref_ctrl,2));
% p_ref_plot = p_ref_ctrl(:, idx_ref_plot);
% v_ref_plot = v_ref_ctrl(:, idx_ref_plot);
% theta_ref_plot = theta_ref_ctrl(:, idx_ref_plot);
% omega_ref_plot = omega_ref_ctrl(:, idx_ref_plot);
% 
% figure('Name','States (MPC vs Reference)','Color',[1 1 1],'Position',[100 100 1000 900]);
% subplot(6,2,1); plot(tvec, p_traj(1,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, p_ref_plot(1,:), 'r--');
% ylabel('p_x'); legend('actual','ref'); grid on;
% subplot(6,2,2); plot(tvec, p_traj(2,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, p_ref_plot(2,:), 'r--');
% ylabel('p_y'); legend('actual','ref'); grid on;
% subplot(6,2,3); plot(tvec, p_traj(3,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, p_ref_plot(3,:), 'r--');
% ylabel('p_z'); legend('actual','ref'); grid on;
% 
% subplot(6,2,4); plot(tvec, v_traj(1,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, v_ref_plot(1,:), 'r--');
% ylabel('v_x'); legend('actual','ref'); grid on;
% subplot(6,2,5); plot(tvec, v_traj(2,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, v_ref_plot(2,:), 'r--');
% ylabel('v_y'); legend('actual','ref'); grid on;
% subplot(6,2,6); plot(tvec, v_traj(3,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, v_ref_plot(3,:), 'r--');
% ylabel('v_z'); legend('actual','ref'); grid on;
% 
% subplot(6,2,7); plot(tvec, theta_traj(1,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, theta_ref_plot(1,:), 'r--');
% ylabel('\theta_x'); legend('actual','ref'); grid on;
% subplot(6,2,8); plot(tvec, theta_traj(2,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, theta_ref_plot(2,:), 'r--');
% ylabel('\theta_y'); legend('actual','ref'); grid on;
% subplot(6,2,9); plot(tvec, theta_traj(3,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, theta_ref_plot(3,:), 'r--');
% ylabel('\theta_z'); legend('actual','ref'); grid on;
% 
% subplot(6,2,10); plot(tvec, omega_traj(1,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, omega_ref_plot(1,:), 'r--');
% ylabel('\omega_x'); legend('actual','ref'); grid on;
% subplot(6,2,11); plot(tvec, omega_traj(2,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, omega_ref_plot(2,:), 'r--');
% ylabel('\omega_y'); legend('actual','ref'); grid on;
% subplot(6,2,12); plot(tvec, omega_traj(3,:), 'b', tvec(1:numel(idx_ref_plot))*Ts, omega_ref_plot(3,:), 'r--');
% ylabel('\omega_z'); legend('actual','ref'); grid on;
% sgtitle('Koopman-MPC Tracking: Actual vs Reference States');
% 
% figure('Name','Inputs (2x2)','Color',[1 1 1],'Position',[200 200 700 500]);
% subplot(2,2,1); plot(tvec, u_traj(1,:)); ylabel('f_t'); grid on;
% subplot(2,2,2); plot(tvec, u_traj(2,:)); ylabel('M_1'); grid on;
% subplot(2,2,3); plot(tvec, u_traj(3,:)); ylabel('M_2'); grid on;
% subplot(2,2,4); plot(tvec, u_traj(4,:)); ylabel('M_3'); grid on;
% sgtitle('Koopman-MPC Inputs (Thrust and Moments)');
% 
% disp('Simulation complete.');