%% Rendering Example

%% Build Circle
outputSize = 100;
frameCenter = ceil(outputSize/2);

numPoints = 40; % Number of points in circle polygon
radius = 5;
[xCirc,yCirc] = cylinder(1,numPoints);
pointsCirc = [xCirc(1,:); yCirc(1,:)]*radius+frameCenter;
totalRadiance = 5; %Completely filled pixel value
superSamplingFactor = 1;
frame = raster_v5(pointsCirc,totalRadiance,outputSize,outputSize,superSamplingFactor);
imshow(frame);