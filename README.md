# Aimpoint-Neural-Network


Object Aim-Point Neural Network Package README file

Creator: Adrian Markelov
email: amarkelo@andrew.cmu.edu


Summary:
This package is a framework being developed to track critical points of objects using a neural network. 

General Steps to Execution:
1. Set up the input parameters for generating data
2. Generate Data with data_gen.m script
3. Set up the input parameters for the neural network
4. Modify present data to plot the specific data you want to see
   (currently it presents the training and test cost functions and
    a few random predictions with there corresponding truths)
5. Run neural_network()


Setting up data_gen:
- Variables to modify
	- image_file: string
              * variable holds the file path to an indexed image that will be used either
			    for training or testing
			  * ‘/file_path/image_%d_%d.mat’
			  * format: _%d_%d corresponds to the way data_gen.m produces data files
			  * 1st %d indexes size of object
			  * 2nd %d indexes angle of object
	- centroids_file: string
			  * variable holds the file path to all of critical points truth data
			    that corresponds to the input images
			  * data: 2xm matrix where m is the size of all data and 2 corresponds to (x,y)
			  * ‘/file_path/feature_truths.mat’ (ex. centroids)
	- num_sizes: integer
			  - number of variations of major axis sizes of object. Must correspond to
			    the same value as in the neural_net() script
	- num_angles: integer
			  - number of evenly distributed angles sampled from 0-360 degrees. Must 
			    correspond to the same value as in the neural_net() script.
	- height_diff: integer
			  - the size the height will change between the generation of each new image
			    for a corresponding angle starting at initial_height
	- height_width_ratio: floating point between 0-1
			  - scaling factor for width given the height
			  - assuming height is major axis of pencil
	- head_ratio: floating point between 0-1
			  - scaling factor for how much of the major axis is used for the angles pointy
			    section of a pencil
	- raster_v5 variables
		> total_radiance: keep at 5 for close to binary image
		> output size: size of a side of the square image being generated
		> super_sampling_factor: keep at 1 I don’t know what this does


- Steps
	1. Create a directory for the images and centroids files
	2. initialize all of the given variables above
	3. run





Setting up neural_network:
- Variables to modify
	- training_state: boolean
			  - true: run training algorithm and write weights to ’neural_network_file’
			  - false: load an already generated neural network weight matrix
	- image_file: string
			  - variable holds the file path to an indexed image that will be used either
			    for training or testing
			  - ‘/file_path/image_%d_%d.mat’
			  - format: _%d_%d corresponds to the way data_gen.m produces data files
			  - 1st %d indexes size of object
			  - 2nd %d indexes angle of object
	- centroids_file: string
			  - variable holds the file path to all of critical points truth data
			    that corresponds to the input images
			  - data: 2xm matrix where m is the size of all data and 2 corresponds to (x,y)
			  - ‘/file_path/feature_truths.mat’ (ex. centroids)
	- neural_network_file: string
			  - variable holds the file path to a saved neural network. either one that
			    already exists or one that is being written to.
			  - ‘/file_path/neural_net.mat’
			  - file contents: 
				- W: weights matrix
				- B: bias weights matrix
				- training_avg_cost_func: cost function for training data
				- test_avg_cost_func: cost function for test data
	- num_sizes: integer
			  - number of variations of major axis sizes of object. Must correspond to
			    the same value as in the data_gen.m script
	- num_angles: integer
			  - number of evenly distributed angles sampled from 0-360 degrees. Must 
			    correspond to the same value as in the data_gen.m script.
	- integral_size: integer
			  - number of evenly distributed area segmentations made of the image along the 
			    x and y axis individually. So a total of 2*integral_size segmentations are made
	- image_size: integer
			  - number of pixels of each axis of the image. As of now it assumes a square image
			    but some adjustments to integrate image and norm_2_position/ position_2_norm
			    could change that
	- topology: array of integers
			  - An array that describes how many neurons are in each layer of the neural network.
			  - layer 1: is ALWAYS 2 * integral_size. Unless you have modified that code for handling
			    2D image input vector
			  - hidden layers: only 1 is needed for relatively accurate results. The size of the input
			    layer should be orders of magnitude smaller than the training data or it will be impossible
			    to learn the network. Ex. training set of 960 image -> 8-10 hidden neurons.
			  - output layer: always 2 neurons (x,y) pixel of the critical point you are searching for.
			  - note: length(topology) = number of layers in the neural network
	- iter: integer
			  - the number of times you want to batch learn over all of the given training data set. Each
			    iteration passes through all of the training data. The meaning of this number will change 
			    if you implement another version of back-prop such of stochastic or mini-batch back-prop
	- alpha: floating point value usually between 0-1
			  - this is the step size of the gradient decent process. deltas will be calculated indicating
			    a gradient direction to minimize error, but alpha tells you how far to step in that direction.
			    Too small and convergence will be very slow or it will get stuck in a very small local minima.
			    Too big and it will never really converge it will just bounce around un-optimal values.
			  - recommended value: .2-.25
			  - improvement: you could add code to make this value smaller after you have started converging.
			    be careful not to do this too soon or you will converge to a local minima
- steps
	1. Initialize all of the variables above. Especially make sure variables overlapping with data_gen have the same value.
	2. If you do not have a pre-saved neural network that you want to run instantly then make sure training state is set to true
	   so that the training algorithm will be run and the results will be saved to the specified file. Other wise, setting training_state
	   to false will not run the training algorithm and will quickly load all of the data and the neural network and run it across all of
	   the data in a few seconds. Remember likely training will take at the very least 20 mins to 1/2 hours for about 1000 images.
	3. setting up data presentation: optional- As of now the pesent_data(…) function plots out some random images with the corresponding 
	   prediction point and truth point, as well as plotting the cost functions of the training and test data. If you would like to sift 
	   through all of the results either place a break point right after present_data is called and plot the points yourself or change the
	   indexes of the plot functions in present_data(…) or finish writing the function save_marked_data(…) at the very bottom of neural_network
	   which is intended to write all of the plotted graphs to file so you may easily sift through them as you like. 
	4. Finally you are ready to run

Separate Helper Code Files

Pencil.m
- Matlab object file representing a pencil.
- Pencil(height, width, angle, head_ratio, frame_center)
	- generates a pencil object for the given parameters
- note: this can easily be replaced with any shape. Change the ratios and sizes
	of pX and pY and the rest of the code should conform

rotate_object.m
- Matlab function file that takes a shape described by the points in the vectors X and Y
  and rotates them by theta degrees radian.
- This function as of now is only being used in the pencil constructor function but can be
  used to rotate any object
- rotation now is only done about the centroid of the object. Some small modifications 
  should be made to rotate about any given points. Just change c to the point you want
  to rotate about.




Future: (Improvements and additions that should be made)

Functional Improvements
- 2d image instead of 2 1D histograms along both axis
	- 1D is more susceptible to false positives
	- Testing on incorrect data
- Training at different points
	- Like the tip
- Adding more randomized features
	- Add a really weird and fuzzy shaped eraser at the back of the pencil
- Translational and/ or size normalization
	- Mean normalize histogram
	- Center and interpolate a 2D image
	- Etc.
- Recurrent neural networks 
	- Feeds the previous estimations and or images to help predict the next position
	- making tracking over video time better

Optimization and Simplification Improvements
- Stochastic or mini-batch gradient descent
	- Much faster training cycles
	- Especially useful when the data sets are bigger and or if the network gets bigger
- GPU parallelization
	- Make run time and training significantly faster
- Tensor flow
	- Fast neural network framework for deep learning
	- Can utilize GPU parallelization
	- Super optimized vector math (more than just Matlab’s cache optimized math)
- Use preprocessed features
	- Instead of using the image itself use a few preprocessed features as inputs to the
	  neural network 
	- Though it may inherit and amplify error of processed data 
