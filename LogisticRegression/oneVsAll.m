function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%Trains multiple logistic regression classifiers and 
%   returns all the classifiers in a matrix all_theta,

%Some useful variables
m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X]; % add ones to the X data matrix

%To use fmincg function
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels,
  [theta] = fmincg(@(t)(costFunction(t, X, (y==c), lambda)), initial_theta, options);
  all_theta(c, :) = theta';
end


end