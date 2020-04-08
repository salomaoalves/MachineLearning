function g = sigmoid(z)
% Compute sigmoid function

% Initialize some useful values
g = zeros(size(z));

g = 1 ./ (1 + exp(-z));

end
