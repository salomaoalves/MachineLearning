function plotData(X, y)
%Plot a graphic with 2 variabels (to Logistic Regression)

figure; hold on;

plot(X(y==1, 1),X(y==1, 2),'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(y==0, 1),X(y==0, 2),'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)

hold off;

end