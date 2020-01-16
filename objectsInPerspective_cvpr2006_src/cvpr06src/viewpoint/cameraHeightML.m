function yc = cameraHeightML(v0, v, h, t, models)
% yc = estimateCameraHeightML(v0, v, h, t)
% Returns the ML estimate of the camera height
% v0 - horizon position (0-1 from bottom)
% v - object positions (0-1 from bottom)
% h - object heights (0-1)
% t - object types (0 = background, 1 = car, ...)
% models - the mean and std for each type of object

h2 = h ./ (v0-v);

mu = [models(t).mu]; mu = mu(:);
sigma = [models(t).sigma]; sigma = sigma(:);
a = sum(h2.^2 ./ sigma.^2);
b = -sum(h2 .* mu ./ sigma.^2);
c = length(v);

yc = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
