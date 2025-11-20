function x_next = rk4_step(x, u, params, dt)
    k1 = quad_dynamics(0, x, u, params);
    k2 = quad_dynamics(0, x + 0.5*dt*k1, u, params);
    k3 = quad_dynamics(0, x + 0.5*dt*k2, u, params);
    k4 = quad_dynamics(0, x + dt*k3, u, params);

    x_next = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6;
end
