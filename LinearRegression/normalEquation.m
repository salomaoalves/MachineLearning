function [theta] = normalEquation(X, y)
% Computes the closed-form solution to linear regression using the normal equations

theta = zeros(size(X, 2), 1); %

theta = ((X' * X)^-1 * X') * y; % do the normal equations

end