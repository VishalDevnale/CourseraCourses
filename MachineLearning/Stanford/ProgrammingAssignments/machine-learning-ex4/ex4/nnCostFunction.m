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


%Binary vectorization of output labels i.e. Y
%new_y.shape = 5000*10
new_y = zeros(m,num_labels);
for iter = 1:m
  new_y(iter,y(iter,1)) = 1;
end

%Adding bias term to input vector. a1.shape = 401*5000
a1 = [ones(m,1) X]';

%Theta1.shape = 25*401, z2.shape = (25*401)*(401*5000) = 25*5000 
z2 = Theta1*a1;

%This is like activtion function. Activation function: Sigmoid (Logistic regression like)
%Here a2.shape = 25*5000 
a2 = sigmoid(z2)';

%Adding bias term to input vector of next layer. a2.shape = 26*5000
a2 = [ones(m,1) a2]';

%Theta2.shape = 10*26, z3.shape = (10*26)*(26*5000) = 10*5000
z3 = Theta2*a2;

%This is like activtion function. Activation function: Sigmoid (Logistic regression like)
%Here a3.shape = 10*5000
a3 = sigmoid(z3)';

%Non-Regularized
%J = sum(sum(-new_y.*log(a3)-(1-new_y).*log(1-a3)))/m;

%Regularized, regularization is not performed on bias term.
Theta1_R = Theta1;
Theta1_R(:,1) = 0;
Theta2_R = Theta2;
Theta2_R(:,1) = 0; 

J = sum(sum(-new_y.*log(a3)-(1-new_y).*log(1-a3)))/m + (lambda/(2*m)) * ( sum(sum(Theta1_R.*Theta1_R)) + sum(sum(Theta2_R.*Theta2_R)) ); 

%Theta grdient

%delta3.shape = (5000*10)' = 10*5000
delta3 = (a3 - new_y)';

% delta2.shape  = (26*10)*(10*5000) = 26*5000
delta2 = (Theta2_R' * delta3);

% delta2.shape = 25*5000 .* 25*5000 = 25*5000
delta2 = delta2(2:end,:).*sigmoidGradient(z2);

Theta1_grad = (1/m)*(delta2*a1') + (lambda/m)*(Theta1_R);
Theta2_grad = (1/m)*(delta3*a2') + (lambda/m)*(Theta2_R); 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
