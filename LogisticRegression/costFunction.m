function [J,grad] = costFunction(theta, X, y, lambda=0)
%Compute cost function for logistic regression with (or not) regularization 

%Initialize some useful values
m = length(y);
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

J = (1/m) * (-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m)) * theta_reg'*theta_reg;

grad(1) = (1/m) * sum((h - y) .* X(:,1));
grad(2) = (1/m) * sum((h - y) .* X(:,2)) + (lambda/m) * theta(2);
grad(3) = (1/m) * sum((h - y) .* X(:,3)) + (lambda/m) * theta(3);
grad(4) = (1/m) * sum((h - y) .* X(:,4)) + (lambda/m) * theta(4);

end