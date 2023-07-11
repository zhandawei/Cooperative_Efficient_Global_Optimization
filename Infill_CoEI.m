function y = Infill_CoEI(x,kriging_model,fmin,best_x,index)
n = size(x,1);
new_x = repmat(best_x,n,1);
new_x(:,index) = x;
[u,s] = Kriging_Predictor(new_x,kriging_model);
y = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
end
