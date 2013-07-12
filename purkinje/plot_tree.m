function plot_tree(N,adj,dist,terminals)


mn = min(dist);
mx = max(dist);

for i = 1:size(adj,1)
   nodes = find(adj(i,:));
   for j  = 1:length(nodes);
      C = [N(i,:); N(nodes(j),:)];
      d1 = dist(i);
      d2 = dist(nodes(j));
      rgb = val2rgb((d1+d2)/2, mn, mx);
      plot3(C(:,1),C(:,2),C(:,3),'color',rgb)
   end
end

idx = find(terminals);

for i = 1:length(idx)

   plot3( N(idx(i),1), N(idx(i),2), N(idx(i),3),'.','markersize',15,'markeredgecolor', val2rgb(dist(idx(i)),mn,mx ));

end
daspect([1,1,1])