function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


temp = ones(1);
theta2 = transpose(theta);
temp2 = zeros(sizeof(y));
yo = sigmoid(X * theta);
temp2 = transpose(y)*log( yo) + transpose(1 - y) * log(1 -  yo);

J = temp * temp2;

J  =  J./m;

J  = -1.*J;

temp =  (X' * (yo - y))./m;
grad = temp;






% =============================================================

end
