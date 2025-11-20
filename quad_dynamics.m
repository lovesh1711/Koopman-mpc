function x_dot=quad_dynamics(t, x,u, params)

m=params.m;
J=params.J;
g=params.g;
e3=[0;0;1];

%unpacking the states
p=x(1:3);
v=x(4:6);
Rvec=x(7:15);
R=reshape(Rvec, 3,3);
omega=x(16:18);

% input
ft=u(1);
M=u(2:4);

% translational dynamics
p_dot=v;
% v_dot=(1/m)*R*[-0;0;ft]-g*e3;

v_dot = R'*(g * e3 - (1 / m) * R * [0;0;ft]); % net accleration


% rotational dynamics
R_dot=R*hat(omega);
omega_dot=J\(M-cross(omega,J*omega));
% omega_dot=J\(M-hat(omega)*J*omega);

x_dot=[p_dot; v_dot; R_dot(:); omega_dot];





end