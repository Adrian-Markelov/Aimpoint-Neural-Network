function [ frame ] = raster_v5( targetPointsFrame,tRadiance, nrows,ncolumns,...
    superSamplingFactor )
% Takes a discrete set of points from a target projected on an aperature
% plane and rasters them onto pixel areas

nrows = nrows*superSamplingFactor;
ncolumns = ncolumns*superSamplingFactor;
targetPointsFrame = targetPointsFrame * superSamplingFactor;

frame = zeros(nrows,ncolumns);

rowMin = floor(min(targetPointsFrame(2,:)));
rowMax = ceil(max(targetPointsFrame(2,:)));

if rowMin < 1
    rowMin = 1;
end
if rowMax > nrows
    rowMax = nrows;
end

colMin = floor(min(targetPointsFrame(1,:)));
colMax = ceil(max(targetPointsFrame(1,:)));

if colMin < 1
    colMin = 1;
end
if colMax > ncolumns
    colMax = ncolumns;
end

[targetX,targetY] = poly2cw(targetPointsFrame(1,:),targetPointsFrame(2,:));

for rowNum = rowMin:rowMax
    for colNum = colMin:colMax
        
        pixelTestX = [colNum-1, colNum-1, colNum, colNum];
        pixelTestY = [rowNum-1, rowNum, rowNum, rowNum-1];
        [intersectX, intersectY] = polybool('intersection',pixelTestX,pixelTestY,...
            targetX,targetY);
        
        frame(rowNum,colNum) = polyarea(intersectX,intersectY);
  
    end
end


if max(max(frame))>0
    frame = frame*tRadiance/superSamplingFactor^2;
end

end

