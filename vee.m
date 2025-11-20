function a = vee(S)
    a=zeros(3,1);
    a(1)=-S(2,3);
    a(2)=S(1,3);
    a(3)=-S(1,2);

    % a = [S(3,2); S(1,3); S(2,1)];
end