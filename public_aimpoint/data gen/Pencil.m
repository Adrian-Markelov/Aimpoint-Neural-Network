%Creator: Adrian Markelov
%Contact: amarkelo@andrew.cmu.edu
%% NOTE
%   - the pX, pY vectors correspond to x,y cordinates of an image
%     starting at the top right corner. To use these as pixel values
%     (row, col) = (Y, X) rounded
%   - X, Y are sub-pixel precise
classdef Pencil
    properties
        pX; % Vector of x axis values
        pY; % Vector of y axis values
        c;  % tuple of (x,y) centroid of pencil
    end
    methods
        function p = Pencil(height, width, angle, head_ratio, frame_center)
            num_points = 5;
            p.pX = [0 height*head_ratio height height*head_ratio 0]+frame_center;
            p.pY = [0 0 width/2 width width]+frame_center;
            [p.pX, p.pY, p.c] = rotate_object(p.pX, p.pY, angle, num_points);
        end
    end
end
