load base
load lv
load rv
load epi

[E,N] = heart;
E = E(:,2:4);
N = N(:,2:4);

i = unique(E(base,:));
x = median(N(i,1));
y = median(N(i,2));
dist = sum(((N(i,1:2) - ones(length(i),1)*[x,y]).^2)');
[val,idx] = min(dist);

root = i(idx);
C = [N(root,1)-0.5, N(root,2), N(root,3)]; 





%[right, dist_rv, term_rv] = get_tree(E,N,rv,C);
[left, dist_lv, term_lv] = get_tree(E,N,lv,C);

size(dist_lv(dist_lv ~=0))
clf; hold on;
%plot_tree(N,right, dist_rv, term_rv)
plot_tree(N,left, dist_lv, term_lv)
