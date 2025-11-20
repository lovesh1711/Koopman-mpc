function psi=observable_phi(x,p)

    p_lin=x(1:3);
    v=x(4:6);
    R=reshape(x(7:15),3,3);
    omega=x(16:18);
    
    % hat-map of imega
    w_hat=hat(omega);
    
    h_stack=[];
    A=w_hat;
    for i=1:p
        Hi=R*A;
        h_stack=[h_stack;Hi(:)];
        A=A*w_hat;
    end
    
    % lifted state
    psi=[p_lin; v; R(:);omega;h_stack];
    
end