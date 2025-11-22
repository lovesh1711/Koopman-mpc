function Theta = so3_log(R)
%SO3_LOG  Matrix logarithm on SO(3)

    [U,~,V] = svd(R);
    R = U*V';

    % compute angle
    cos_phi = (trace(R) - 1)/2;
    cos_phi = max(min(cos_phi,1),-1);  % clamp for numerical safety
    phi = acos(cos_phi);

    if abs(phi) < 1e-6
        % small-angle approximation
        Theta = 0.5*(R - R');
    else
        Theta = (phi/(2*sin(phi))) * (R - R');
    end
end


% function Theta = so3_log(Rin)
% % Robust matrix logarithm on SO(3)
% % returns Theta in so(3) such that expm(Theta) = R (approximately)
% 
%     % Reproject to SO(3) robustly using SVD + D
%     [U,~,V] = svd(Rin);
%     D = eye(3);
%     if det(U*V') < 0
%         D(3,3) = -1;
%     end
%     R = U * D * V';
% 
%     % compute angle (clamp for numerical safety)
%     cos_phi = (trace(R) - 1) / 2;
%     cos_phi = max(min(cos_phi, 1), -1);
%     phi = acos(cos_phi);
% 
%     eps_small = 1e-8;
%     eps_pi    = 1e-6;
% 
%     if phi < eps_small
%         % small angle: use first-order approximation
%         Theta = 0.5 * (R - R');
%         return;
%     end
% 
%     if abs(pi - phi) < eps_pi
%         % angle near pi: numerically delicate. Extract rotation axis robustly.
%         % From: (R + I) = 2 * (axis * axis^T) when phi = pi.
%         % Find the column of (R + I) with largest norm:
%         S = (R + eye(3)) / 2;
%         [~, idx] = max(diag(S));
%         v = S(:, idx);
%         if norm(v) < 1e-6
%             % fallback
%             Theta = (phi / (2*sin(phi))) * (R - R');
%             return;
%         end
%         axis = v / norm(v);
%         % build skew matrix for axis * phi
%         Theta = hat(axis * phi);
%         return;
%     end
% 
%     % general case
%     Theta = (phi / (2*sin(phi))) * (R - R');
% end
