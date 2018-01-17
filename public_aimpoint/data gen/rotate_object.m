% Author: Adrian Markelov
% Contact: amarkelo@andrew.cmu.edu

%% rotate_object
% This function take in an object defined by num_points number of critical
% points of an object and two vector X and Y describing its corresponding
% coordinates and it will rotate the image by theta degrees in radians.
% As of now to rotates the object by its centroid but to change this
% change c to the desired rotation point
function [X_new, Y_new, centroid] = rotate_object(X, Y, theta, num_points)
    P = [X; Y]; % matrix of all the points of the object
    c = [sum(X); sum(Y)]/num_points; % centroid of the object
    C = repmat(c, [1 num_points]); % centroid matrix
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)]; %standard rotation matrix
    P_new = R*(P-C) + C; % center object - rotate about origin - replace object offset
    X_new = P_new(1,:);
    Y_new = P_new(2,:);
    centroid = c;
end