class Graph
        def initialize(properties, types, relationships)
                @graph = {}
                @node_lists = {}
                @types = types
                @relationships = relationships
                @node_count = properties[:node_count]
                @max_edge_count  = properties[:max_edge_count]
                @source_vertex = 0
                @total_edge_count = 0
                generate_node_types && generate_edges
        end

        
        def print_edge_counts
                total_count  = 0
                @node_count.times do |current_node|
                puts "#{total_count} #{@graph[current_node][:connections].size} #{@graph[current_node][:type]}"
                total_count += @graph[current_node][:connections].size
                end
        end

        def print_edges
                @node_count.times do |current_node|
                @graph[current_node][:connections].each do |destination|
                puts "#{destination} #{rand(10)}" #{@graph[current_node][:type]}=>#{@graph[destination][:type]}"
        	        end
                end
        end
         
        def print_node_count
                p(@node_count)
        end

        def print_source_vertex
                p(@source_vertex)
        end

        def print_total_edge_count
                p(@total_edge_count)
        end
       
        private
        def generate_node_types
                @node_count.times do |current_node|
                        @graph[current_node] = {}
                        type =  @types.sample
                        @graph[current_node][:type] = type
                        @node_lists[type] ||= []
                        @node_lists[type] << current_node
                end
        end

        def generate_edges
                @node_count.times do |current_node|
                        edge_count_for_current_node = 1 + rand(@max_edge_count - 1)
                        @graph[current_node][:connections] = []
                        generate_edges_for(current_node, edge_count_for_current_node) unless @relationships[@graph[current_node][:type]].nil?
                end
        end

        def generate_edges_for(current_node, count)
                p @node_lists
                count.times do
                        relationship  = @relationships[@graph[current_node][:type]].sample
                        destination_node = @node_lists[relationship].sample
                        p relationship
                        p destination_node
                        while(@graph[current_node][:connections].include?(destination_node) || destination_node == current_node) do
                                destination_node = @node_lists[relationship].sample
                        end	
                        @graph[current_node][:connections] << destination_node
                        @total_edge_count += count
                end
        end
end

g = Graph.new({:node_count => 10000000, :max_edge_count => 100}, ["person", "restaurant"], {"person" => ["restaurant", "person"]} )
g.print_node_count
g.print_edge_counts

puts
g.print_source_vertex
puts

g.print_total_edge_count
g.print_edges

