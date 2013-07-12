function c = val2rgb(v, mn,mx)



if v<mn
 c=[0,0,0.5];
 return
end
if v>mx
 c=[0.5,0,0];
 return
end

 
val = 0.5+4*((v - mn)/(mx-mn)).^0.95; % 0.95 centers green better than 1
red =    max(0,min(1,min(val-2,5-val)));
green =  max(0,min(1,min(val-1,4-val)));
blue =   max(0,min(1,min(val-0,3-val)));

c = [red green blue];