node_count = 1000
max_edge_count = 1000
puts(node_count)
graph = {}
total_count = 0
node_count.times do |current_node|

	edge_count_for_current_node = 1 + rand(max_edge_count - 1)

	puts "#{total_count} #{edge_count_for_current_node}"
	total_count += edge_count_for_current_node

	graph[current_node] = [] 
	edge_count_for_current_node.times do
		destination_node = rand(node_count)
		while(graph[current_node].include?(destination_node) || destination_node == current_node) do
			destination_node = rand(node_count)
		end	
		graph[current_node] << destination_node
	end
end

#puts "\n#{rand(node_count)}\n"
puts "\n0\n"
puts
puts total_count
#p graph
node_count.times do |current_node|
	graph[current_node].each do |destination|
		puts "#{destination} #{rand(10)}"
	end
end

p "Total Edges -- #{total_count}"
