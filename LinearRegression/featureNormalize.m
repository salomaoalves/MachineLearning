function [X_norm, mu, sigma] = featureNormalize(X)
%Normalizes the features in X

%Initialize some useful values
X_norm = X;
m = size(X, 2);
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

for i = 1:m %for each example
  mu(i) = mean(X(:,i)); %mean
  sigma(i) = std(X(:,i)); %standard deviation
  X_norm(:,i) = (X_norm(:,i)-mu(i)) / sigma(i); %X ith normalized
end

end