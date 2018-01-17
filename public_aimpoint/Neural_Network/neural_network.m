%% TODO
%   - function: integrate_image_2d() to replace the 2-axis 1d histogram
%   - 
%   - 

%GOOD RESOURCE
%   - http://adventuresinmachinelearning.com/neural-networks-tutorial/

%% Main Routine
function neural_network()
    %Decide weather you want to train and generate new weights or just
    %load pre-existing weights from a file
    training_state = true;
    
    %File IO
    image_file = '/Users/markeas1/Desktop/Winter_2017-18/data/gen_images_1/pencil_image_%d_%d.mat';
    centroids_file = '/Users/markeas1/Desktop/Winter_2017-18/data/gen_images_1/centroids.mat';
    neural_network_file = '/Users/markeas1/Desktop/Winter_2017-18/saved_models/tests/net_2/neural_net_2.mat';

    %Rocket Variables
    num_sizes = 20;
    num_angles = 72;
    integral_size = 60;
    image_size = 300;

    %Neural Network Variables
    topology = [integral_size*2 10 2]; %network topology
    iter = 12000; %iterations of learning
    alpha = .25; % learning rate
    n_layers = length(topology);
    
    %Make these variables visible outside of the scope of the if statements
    training_data = {};
    test_data = {};
    test_images = [];
    
    if(training_state)
        %split data into a training and test set
        [training_data, test_data, test_images] = get_data(image_file, centroids_file, num_sizes, num_angles, integral_size, image_size);

        %train the neural network using back propagation 
        [W, B, training_avg_cost_func, test_avg_cost_func] = train_nn(topology, training_data, test_data, iter, alpha);
        
        %save the neural network for future use
        save_neural_network_file(neural_network_file, W, B, training_avg_cost_func, test_avg_cost_func);
    else
        %use all data as test data
        [training_data, test_data, test_images] = get_data(image_file, centroids_file, num_sizes, num_angles, integral_size, image_size);
        
        %test_data{1} = [test_data{1},training_data{1}];
        %test_data{2} = [test_data{2},training_data{2}];
        
        %load the existing neural network
        [W, B, training_avg_cost_func, test_avg_cost_func] = load_neural_network_file(neural_network_file);
    end
    t = cputime;
    %using the trained neural network weights predict the test data
    y_pred = predict_point(W, B, test_data, n_layers);
    e = cputime-t;
    disp('time: ');
    disp(e/length(test_data{1}(1,:)));
    %Compare the predicted results with the actual results
    y_true = test_data{2};
    RMSE = sqrt(mean((y_true-y_pred).^2));
         
    % Convert the predicted output back into a pixel value position
    y_pred_position = norm_2_position(y_pred, image_size);
    y_true_position = norm_2_position(y_true, image_size);
    
    %Plot and print out the data
    present_data(y_pred_position, y_true_position, test_images, training_avg_cost_func, test_avg_cost_func, RMSE)
    
    
end


%% FILE-IO AND DATA ACCESSING HELPER FUNCTIONS BELOW
%--------------------------------------------------------------------------
%-----------------   FILE-IO AND DATA ACCESSING CODE   --------------------
%--------------------------------------------------------------------------


%% present_data(y_pred_position, test_images)
% Display the result of training with the test data
function present_data(y_pred_position, y_true_position, test_images, avg_cost_func, test_avg_cost_func, RMSE)
    disp('RMSE: ');
    disp(mean(RMSE));
    
    %PLOT THE DATA
    hold on;
    %display results
    figure(1);
    plot(avg_cost_func);
    plot(test_avg_cost_func);
    
    figure(2);
    imagesc(test_images(:,:,1));
    hold on;
    plot(y_pred_position(1,1), y_pred_position(2,1),'r.','markersize',20);
    plot(y_true_position(1,1), y_true_position(2,1),'g.','markersize',20);
    
    figure(3);
    imagesc(test_images(:,:,100));
    hold on;
    plot(y_pred_position(1,100), y_pred_position(2,100),'r.','markersize',20);
    plot(y_true_position(1,100), y_true_position(2,100),'g.','markersize',20);
    
    figure(4);
    imagesc(test_images(:,:,300));
    hold on;
    plot(y_pred_position(1,300), y_pred_position(2,300),'r.','markersize',20);
    plot(y_true_position(1,300), y_true_position(2,300),'g.','markersize',20);
    
    figure(5);
    imagesc(test_images(:,:,400));
    hold on;
    plot(y_pred_position(1,400), y_pred_position(2,400),'r.','markersize',20);
    plot(y_true_position(1,400), y_true_position(2,400),'g.','markersize',20);
end

%% save_neural_network_file(neural_network_file, W, B, avg_cost_func)
% Save neccessary neural network variables to a file
function save_neural_network_file(neural_network_file, W, B, training_avg_cost_func, test_avg_cost_func)
    save(neural_network_file, 'W', 'B', 'training_avg_cost_func', 'test_avg_cost_func');
end

%% load_neural_network_file(neural_network_file)
%pull out vars from file to set up new neural network
function [W, B, training_avg_cost_func, test_avg_cost_func] = load_neural_network_file(neural_network_file)
    network_file_obj = load(neural_network_file);
    W = network_file_obj.W;
    B = network_file_obj.B;
    training_avg_cost_func = network_file_obj.training_avg_cost_func;
    test_avg_cost_func = network_file_obj.test_avg_cost_func;
end

%% get_data(image_file, centroids_file, num_sizes, num_angles, integral_size, image_size)
% Splitting Data:
%   Every 2 consecuative rockets will be used for training and 
%   every third will be split into testing data
% data format: [c = centroid]
%   { image image image ... image;
%     c     c     c     ... c     }
function [training_data, test_data, test_images] = get_data(image_file, centroids_file, num_sizes, num_angles, integral_size, image_size)
    %load target centroid
    centroid_file_obj = load(centroids_file);
    Centroids = centroid_file_obj.Centroids;
    
    %data set partition sizes
    training_data_size = num_sizes*num_angles*2/3;
    test_data_size = num_sizes*num_angles/3;
    
    %image integral vector
    integral_vector_size = 2*integral_size;
    
    %Initialize data set variables
    training_data_inputs = zeros(integral_vector_size, training_data_size);
    training_data_outputs = zeros(2, training_data_size);
    training_data = {training_data_inputs, training_data_outputs};
    test_data_inputs = zeros(integral_vector_size, test_data_size);
    test_data_outputs = zeros(2, test_data_size);
    test_data = {test_data_inputs, test_data_outputs};
    
    % save test images to mark later
    test_images = zeros(image_size, image_size, test_data_size);
    
    % Tracking partitions of data sets variables
    count = 1;
    training_count = 1;
    test_count = 1;
    
    
    %loop through all image and centroid sets
    for i = 1:num_sizes
        image_angle_index = 1:num_angles;
        image_angle_index = image_angle_index(randperm(length(image_angle_index)));
        for j = 1:num_angles
            %load image
            im_file_obj = load(sprintf(image_file,i, image_angle_index(j)));
            image = im_file_obj.frame;
            
            %get corresponding centroid
            c = [Centroids(i,image_angle_index(j),1);Centroids(i,image_angle_index(j),2)];
            
            %partition to training data
            if(mod(count,3) ~= 0)
                training_data{1}(:,training_count) = integrate_image(image, integral_size);
                training_data{2}(:,training_count) = position_2_norm(c, image_size);
                training_count = training_count + 1;
            %partition to test data
            else
                test_images(:,:,test_count) = image;
                test_data{1}(:,test_count) = integrate_image(image, integral_size);
                test_data{2}(:,test_count) = position_2_norm(c, image_size);
                test_count = test_count + 1;
            end
            count = count + 1;
        end
    end
end


%% IMAGE PREPROCCESSING HELPER FUNCTIONS BELOW
%--------------------------------------------------------------------------
%--------------------   IMAGE PREPROCCESSING CODE   -----------------------
%--------------------------------------------------------------------------

%Integrates and image along the x and y axi
% v[1:integral_size] - x wise integrations
% v[integral_size+1:2*integral_size] - y wise integrations
%images are split up into segments that are integrated 
function v = integrate_image(image, integral_size)
    segment_size = length(image(1,:))/integral_size;
    v = zeros(2*integral_size, 1);
    pixel_index = 1;
    for i = 1:integral_size
        for j = 1:segment_size;
            v(i) = v(i) + sum(image(:,pixel_index));
            v(integral_size+i) = v(integral_size+i) + sum(image(pixel_index,:));
            pixel_index = pixel_index + 1;
        end
    end
end

%convert a centroid x, y tuple to a normalized state
function y_norm = position_2_norm(y, image_size)
    y_norm = y/image_size;
end

%convert a normalized position vector to its original position value
function y = norm_2_position(y_norm, image_size)
    y = y_norm*image_size;
end


%% NEURAL NETWORK HELPER FUNCTIONS BELOW
%--------------------------------------------------------------------------
%------------------------   NEURAL NETWORK CODE   -------------------------
%--------------------------------------------------------------------------

%% gen_weights(topology)
%Initialize all the weights of the neural network
% weights format: Wlij: layer l - from neuron j -> to neuron i (l+1)
%   [ w11  - current -  w1n;
%      |                 | ;
%      next layer        | ;
%      |                 | ;
%     wn1  -        -   wnn ]
function [W, B] = gen_weights(topology)
    len = length(topology)-1;
    W = cell(1,len);
    B = cell(1,len);
    for layer = 1:len
        W{layer} = randn(topology(layer+1), topology(layer));
        B{layer} = randn(topology(layer+1), 1);
    end
end

%% gen_delta_weights(topology)
% This generates the weight differential matrix which is the same as
% gen_weights but instead of being standard normalized random values
% every matrix is initialized with zeros.
%       -  partial(cost)/partial(W) = delta_next * h_l'
%       -  delta_W_l = delta_W_l + partial(cost)/partial(W)
function [delta_W, delta_B] = gen_delta_weights(topology)
    len = length(topology)-1;
    delta_W = cell(1,len);
    delta_B = cell(1,len);
    for layer = 1:len
        delta_W{layer} = zeros(topology(layer+1), topology(layer));
        delta_B{layer} = zeros(topology(layer+1), 1);
    end
end

%% act_f(z)
% Activation Function: Sigmoid
function h = act_f(z)
    h = 1./(1 + exp(-z));
end

%% act_f_deriv(z)
% Derivative of Activation Function: Sigmoid'
function h_prime = act_f_deriv(z)
    h_prime = act_f(z).*(1-act_f(z));
end

%% feed_forward(x, W, B) 
% One pass through the neural network of given input vector x
%   - LOG: h and z for all layers of network (we reuse them in back prop)
%   - h = sum of products of layers
%   - z = activation_function(h)
function [h, z] = feed_forward(x, W, B)
    h = {x};
    z = {0};
    for l = 1:length(W)
        if(l == 1)
            layer = x;
        else
            layer = h{l};
        end
        z{l+1} = W{l} * layer + B{l};
        h{l+1} = act_f(z{l+1});
    end
end

%% output_layer_delta(y, h_out, z_out)
% calculate output layer delta (has no forward weights)
% delta = -(y-h).*f'(z)
function delta = output_layer_delta(y, h_out, z_out)
    delta = -1*(y-h_out).*act_f_deriv(z_out);
end

%% hidden_layer_delta(delta_next, w_l, z_l)
% Calculate hidden layer deltas using forward layer delta 
% delta_l = (W_l'*delta_next).*f'(z_l)
function delta = hidden_layer_delta(delta_next, w_l, z_l)
    delta = w_l' * delta_next .* act_f_deriv(z_l);
end

%% train_nn(topology, training_data, iter, alpha)
% Given a Neural Network Topology and training data use Back-PROPOGATION
% to optimize the weights for such NN
function [W, B, training_avg_cost_func, test_avg_cost_func] = train_nn(topology, training_data, test_data, iter, alpha)
    %Seperate training inputs and outputs
    X = training_data{1};
    y = training_data{2};
    X_test = test_data{1};
    y_test = test_data{2};
    
    %Weight and Bias Weight Matric Arrays
    [W,B] = gen_weights(topology);
    n_layers = length(topology);
    
    %size of training and test data
    training_data_size = length(y);
    test_data_size = length(y_test);
    
    %track the cost for each iterations generated weight matrix
    training_avg_cost_func = zeros(iter,1);
    test_avg_cost_func = zeros(iter,1);
    %track learning iteration
    count = 1;
    % loop for iter iterations of learning or (add later convergence)
    while(count <= iter)   
        %unnormalized weights differential (gradient) matrix
        [delta_W, delta_B] = gen_delta_weights(topology);
        training_avg_cost = 0;
        test_avg_cost = 0;
        % Iterate through all of training data
        for i = 1:training_data_size
            delta = {};
            % Pass input vector through net all log all layers outputs
            [h, z] = feed_forward(X(:,i), W, B);
            
            % Propagate backwards through the layers of the network
            % and calculate the weight matrix deltas
            for l = n_layers:-1:1
                %calculate output layer delta (has no forwards weights)
                if(l == n_layers)
                    %delta = -(y-h).*f'(z)
                    delta{l} = output_layer_delta(y(:,i), h{l}, z{l});
                    %Accumulate Cost
                    training_avg_cost = training_avg_cost + norm(y(:,i)-h{l});
                else
                    % Calculate hidden layer deltas using forward layer delta: 
                    % delta_l = (W_l'*delta_next).*f'(z_l)
                    if(l > 1)
                        delta{l} = hidden_layer_delta(delta{l+1}, W{l}, z{l});
                    end
                    % Weight update differential vector
                    % partial(cost)/partial(W) = delta_next * h_l'
                    % delta_W_l = delta_W_l + partial(cost)/partial(W)
                    delta_W{l} = delta_W{l} + delta{l+1} * h{l}';
                    % Bias weight update differential vector
                    delta_B{l} = delta_B{l} + delta{l+1};
                end
            end
        end
        %Log how each iterations Weight matrix performs on the test_data
        for i = 1:test_data_size
            % Pass input vector through net all log all layers outputs
            [h,] = feed_forward(X_test(:,i), W, B);
            %accumulate cost
            test_avg_cost = test_avg_cost + norm(y_test(:,i)-h{n_layers});
        end
        
        % Adjust weights matrix based on one full pass through all training
        % Data by a learning constant of alpha
        for l = length(topology)-1:-1:1
            W{l} = W{l} - alpha * (delta_W{l}/training_data_size);
            B{l} = B{l} - alpha * (delta_B{l}/training_data_size);
        end
        %normalize and update the cost data
        training_avg_cost = training_avg_cost/training_data_size;
        test_avg_cost = test_avg_cost/test_data_size;
        training_avg_cost_func(count) = training_avg_cost;
        test_avg_cost_func(count) = test_avg_cost;
        count = count + 1;
    end
end

%% predict_point(W, B, test_data, n_layers)
% Given an already training weights matrix and test data pass the given
% data through the Network and return the results
function y_pred = predict_point(W, B, test_data, n_layers)
    X = test_data{1};
    m = length(X(1,:));
    y_pred = zeros(2, m);
    for i = 1:m
        [h,] = feed_forward(X(:,i), W, B);
        y_pred(:,i) = h{n_layers};
    end
end







%% BACK BURNER HELPER FUNCTIONS BELOW
%--------------------------------------------------------------------------
%--------------------------   BACK BURNER CODE   --------------------------
%--------------------------------------------------------------------------
% THIS IS UNFINISHED CODE AND NOT WORKING YET

%% save_marked_data(test_images, y_pred_position, num_size, num_angles)
% save test data marked
function save_marked_data(test_images, y_pred_position, num_size, num_angles)
    for i = 1:num_sizes*num_angles/3
        pos = round(y_pred_position(:,i));
        x = cast(pos(1), 'int16');
        y = cast(pos(2), 'int16');
        image = cast(test_images(:,:,i), 'uint8');
        marked_image = insertMarker(image,[x y]); %WE DON'T HAVE THIS FUNCTION
        save(sprintf(marked_images_file, i), 'marked_image');
    end
end








