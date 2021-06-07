function a = ComputeEstimatedCoefficients(x,y,NumberOfCoeff)

PowerElement    = 0:NumberOfCoeff-1;
Mat_x          = repmat(x,1,length(PowerElement));
X               = Mat_x.^repmat(PowerElement, length(x), 1);

a               = (X.'*X)\X.'*y;
end