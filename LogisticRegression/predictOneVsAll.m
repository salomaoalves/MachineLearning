function p = predictOneVsAll(all_theta, X)
%Predict the label for a trained one-vs-all classifier

%Some useful variables
m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X]; %add ones to the X data matrix

[_,p] = max(X*all_theta',[],2);

end