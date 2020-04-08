function J = costFunctionLogisticR(theta, X, y, lambda)
% Compute cost and gradient for logistic regression with (or not) regularization 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

J = (1/m) * (-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m)) * theta_reg'*theta_reg;

end