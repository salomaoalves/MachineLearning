function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Do Gradient Descent to Linear Regression (Simple or Multiple)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
thetaTemp = zeros(size(theta));
sizetheta = length(theta);

for iter = 1:num_iters % repeat until converge

  for i = 1:sizetheta % for each theta
    thetaTemp(i) = theta(i) - alpha*(1/m)*sum((X*theta-y).*X(:,i));
  end;
  theta = thetaTemp(:);

  J_history(iter) = computeCostLR(X, y, theta); % save the cost J in every iteration

end

end