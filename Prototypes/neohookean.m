F = sym('F', [2,2]);
assume(F, 'real');
%norm(F, "fro")^2
F2 = transpose(F)
F
trace(F*F2)
% syms mu lambda J;
% log_J = log(J);
% 
% psi = mu/2 * (norm(F)^2 - 3) - mu * log_J + lambda/2 * (log_J)^2;
% gradient(psi, F(:))