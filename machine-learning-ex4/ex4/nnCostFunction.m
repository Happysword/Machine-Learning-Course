function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%perform FeedForward
X = [ ones(size(X,1),1) ,X];    % 5000x401
Z1 = X*Theta1';
h1 = sigmoid(Z1);     % 5000x25
h1 = [ ones(size(h1,1),1) ,h1]; % 5000x26
h2 = sigmoid( h1 * Theta2');    % 5000x10

%calculate y for oneVSall
yMat = zeros(size(y,1),num_labels ); % 5000x10
for i = 1:num_labels
  yMat(:,i) = (y == i); 
endfor

%calculate cost function with all y values and last theta.
J = 0;
for i = 1:num_labels
    J += (-1/m) * ( yMat(:,i)'*log(h2(:,i)) + (1-yMat(:,i))'*log(1-h2(:,i)) );
end

%regularization
tempTheta1 = Theta1(:,2:end).^2;
tempTheta2 = Theta2(:,2:end).^2;
J = J + (  lambda / (2*m) * ( sum(sum(tempTheta1)) + sum(sum(tempTheta2)) ) );


%grad Calculations
del3 = h2 - yMat; % 5000x10
del2 = sigmoidGradient([ones( size(Z1,1),1 ) Z1]) .* (del3 * Theta2);
del2 = del2(:,2:end);

Delta1 = del2' * X; %10x401
Delta2 = del3' * [ones(size(Z1, 1), 1) sigmoid(Z1)];

%regularization
Theta1_grad = Delta1/m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta2/m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
