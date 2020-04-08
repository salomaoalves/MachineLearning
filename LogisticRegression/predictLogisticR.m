function p = predictLogisticR(theta, X)
% Make prediction to Logistic Regressions models

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

p = sigmoid(X*theta);
p(p >= 0.5) = 1;
p(p < 0.5) = 0;

end
