%% Author: Adrian Markelov
%% Contact: amarkelo@andrew.cmu.edu

%% NOTE
%   - the rX, rY vectors correspond to x,y cordinates of an image
%     starting at the top right corner. To use these as pixel values
%     (row, col) = (Y, X) rounded
%   - X, Y are sub-pixel precise
classdef Rocket
    properties
        rX;
        rY;
        c;
    end
    methods
        function r = Rocket(height, width, angle, head_ratio, frame_center)
            num_points = 5;
            r.rX = [0 height*head_ratio height height*head_ratio 0]+frame_center;
            r.rY = [0 0 width/2 width width]+frame_center;
            [r.rX, r.rY, r.c] = rotate_object(r.rX, r.rY, angle, num_points);
        end
    end
end

