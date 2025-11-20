function [H, G, Aineq, bineq, lb, ub] = build_mpc(Aqp, Bqp, Qtilde, Rbar, z0, Y, u_lb_step, u_ub_step, Nh)
% BUILD_MPC Build QP matrices H, G and inequality constraints for lifted Koopman MPC
%
% Inputs:
%   Aqp, Bqp    - stacked predictive matrices (Nz*Nh x Nz) and (Nz*Nh x nu*Nh)
%   Qtilde      - single-step lifted penalty matrix (Nz x Nz) with Qx in top-left
%   Rbar        - control penalty over horizon (nu*Nh x nu*Nh)
%   z0          - current lifted state (Nz x 1)
%   Y           - stacked reference lifted states over horizon (Nz*Nh x 1)
%   u_lb_step   - lower bound on one-step control (nu x 1)
%   u_ub_step   - upper bound on one-step control (nu x 1)
%   Nh          - prediction horizon
%
% Outputs:
%   H, G        - QP matrices for cost 0.5 U'HU + U' G
%   Aineq, bineq- inequality matrices Aineq * U <= bineq (stacked upper/lower bounds)
%   lb, ub      - vector lower/upper bounds for use if desired (nu*Nh vectors)

% sizes
NzNh = size(Aqp,1);
nuNh = size(Bqp,2);
nu = length(u_lb_step);

% big Qbar is block diag of Qtilde across horizon
Qbar = kron(eye(Nh), Qtilde);

% cost matrices per paper (note factor 2 in paper formulas)
H = 2 * (Bqp' * Qbar * Bqp + Rbar);
% H=0.5*(H+H');
G = 2 * (Bqp' * Qbar * (Aqp * z0 - Y));

% inequality constraints: encode u_lb <= u <= u_ub as [I; -I] * U <= [u_ub; -u_lb]
U_ub = repmat(u_ub_step, Nh, 1);
U_lb = repmat(u_lb_step, Nh, 1);

Aineq = [eye(nuNh); -eye(nuNh)];
% bineq = [U_ub; -U_lb];
bineq = [-U_lb; U_ub];

% also return lb and ub vectors (quadprog supports lb, ub)
lb = U_lb;
ub = U_ub;

end
