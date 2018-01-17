%Creator: Adrian Markelov
%Contact: amarkelo@andrew.cmu.edu

%Object major axis Ranges 50 - 150 inc by 5
%object aimpoint = centroid
%angle 0 - 360

%file io
image_file = '/Users/markeas1/Desktop/Winter_2017-18/data/gen_images_1/pencil_image_%d_%d.mat';
centroids_file = '/Users/markeas1/Desktop/Winter_2017-18/data/gen_images_1/centroids.mat';

%sampling variables
num_sizes = 20;
num_angles = 72;
angle_diff = 2*pi/num_angles;
height_diff = 5;

%Rocket variables
height_width_ratio = 1/8;
initial_height = 50;
height = initial_height;
width = height * height_width_ratio;
angle = 0;
head_ratio = 5/6;

%centroid
Centroids = zeros(num_sizes, num_angles, 2);

%raster_v5 variables
total_radiance = 5;
output_size = 300;
super_samping_factor = 1;
frame_center = output_size/2;

count = 1;
%Rocket sizes
for i = 1:num_sizes
    angle = 0;
    for j = 1:num_angles
        p = Pencil(height, width, angle, head_ratio, frame_center);
        pencil_points = [p.pX; p.pY];
        if(true)
            frame = raster_v5(pencil_points,total_radiance,output_size,output_size,super_samping_factor);
            save(sprintf(image_file, i, j), 'frame');
            Centroids(i,j,:) = p.c;
        end
        disp(count);
        count = count + 1;
        angle = angle + angle_diff;
    end
    height = height + height_diff;
    width = height * height_width_ratio;
end
save(centroids_file, 'Centroids');




