function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% J = 1/m * SUM((-y * log(h(x))) - ((1 - y) * log(1 - h(x)))) + lambda/(2*m) * SUM(thetaj^2)

h = sigmoid(X * theta);

thetaj = theta;
thetaj(1) = 0;

lambdaCost = (lambda / (2 * m)) * sum(thetaj .^ 2);
J = (1 / m) * ((-y' * log(h)) - ((1 - y') * log(1 - h))) + lambdaCost;

% gradient = 1/m * SUM((h(x) - y) * xj) + ((L/m) * thetaj);
lambdaGrad = (lambda / m) * thetaj;
grad = ((1 / m) * (X' * (h - y))) + lambdaGrad;

% =============================================================

end
