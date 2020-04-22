function [pred] = predict(X, theta, mu=0, sigma=1)
% Make prediction to Linear Regressions models

% ATTENTION:
% If you normalize your data to train the model, 
% you need to pass the mu and sigma, 
% if you didn't, don't pass anything


% If you normalized the data
x1 = (1650-mu(1)) / sigma(1);
x2 = (3-mu(2)) / sigma(2);

if size(X') == size(theta)
  predict1 = X * theta; %make prediction
else
  fprintf("The demisions of X or theta is wrong, X->1xm and theta->mx1")
end

end