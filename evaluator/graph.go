package evaluator

import (
	"encoding/json"
	"fmt"
	"sort"
)

// Node is a computation graph node with metadata needed for evaluation.
type Node struct {
	Name      string
	Op        string
	Level     int
	Depth     int
	Shape     map[string][]int
	ConfigRaw json.RawMessage
	BlobRefs  map[string]int
}

// Graph is the computation graph with topological ordering and reverse adjacency.
type Graph struct {
	Input  string
	Output string
	Nodes  map[string]*Node
	Order  []string
	Inputs map[string][]string // node name -> list of predecessor node names
}

// buildGraph constructs a Graph from a parsed CompiledHeader.
// It indexes nodes by name, builds a reverse adjacency map (Inputs),
// and computes a topological sort via Kahn's algorithm.
func buildGraph(header *CompiledHeader) (*Graph, error) {
	g := &Graph{
		Input:  header.Graph.Input,
		Output: header.Graph.Output,
		Nodes:  make(map[string]*Node, len(header.Graph.Nodes)),
		Inputs: make(map[string][]string),
	}

	// Index nodes by name.
	for i := range header.Graph.Nodes {
		hn := &header.Graph.Nodes[i]
		if _, exists := g.Nodes[hn.Name]; exists {
			return nil, fmt.Errorf("duplicate node name: %q", hn.Name)
		}
		g.Nodes[hn.Name] = &Node{
			Name:      hn.Name,
			Op:        hn.Op,
			Level:     hn.Level,
			Depth:     hn.Depth,
			Shape:     hn.Shape,
			ConfigRaw: hn.Config,
			BlobRefs:  hn.BlobRefs,
		}
	}

	// Validate Input and Output nodes exist.
	if _, ok := g.Nodes[g.Input]; !ok {
		return nil, fmt.Errorf("input node %q not found in graph", g.Input)
	}
	if _, ok := g.Nodes[g.Output]; !ok {
		return nil, fmt.Errorf("output node %q not found in graph", g.Output)
	}

	// Build reverse adjacency (Inputs) and in-degree count from edges.
	inDegree := make(map[string]int, len(g.Nodes))
	for name := range g.Nodes {
		inDegree[name] = 0
	}

	for _, edge := range header.Graph.Edges {
		if _, ok := g.Nodes[edge.Src]; !ok {
			return nil, fmt.Errorf("edge references nonexistent source node %q", edge.Src)
		}
		if _, ok := g.Nodes[edge.Dst]; !ok {
			return nil, fmt.Errorf("edge references nonexistent destination node %q", edge.Dst)
		}
		g.Inputs[edge.Dst] = append(g.Inputs[edge.Dst], edge.Src)
		inDegree[edge.Dst]++
	}

	// Validate: input node has no incoming edges.
	if inDegree[g.Input] != 0 {
		return nil, fmt.Errorf("input node %q has %d incoming edges, expected 0", g.Input, inDegree[g.Input])
	}

	// Build forward adjacency list for efficient Kahn's algorithm.
	forward := make(map[string][]string)
	for _, edge := range header.Graph.Edges {
		forward[edge.Src] = append(forward[edge.Src], edge.Dst)
	}

	// Kahn's algorithm for topological sort.
	// Sort initial queue for deterministic order.
	queue := make([]string, 0)
	for name, deg := range inDegree {
		if deg == 0 {
			queue = append(queue, name)
		}
	}
	sort.Strings(queue)

	order := make([]string, 0, len(g.Nodes))
	for len(queue) > 0 {
		// Pop front.
		curr := queue[0]
		queue = queue[1:]
		order = append(order, curr)

		// Decrease in-degree for all successors of curr.
		successors := forward[curr]
		sort.Strings(successors)
		for _, dst := range successors {
			inDegree[dst]--
			if inDegree[dst] == 0 {
				queue = append(queue, dst)
			}
		}
	}

	if len(order) != len(g.Nodes) {
		return nil, fmt.Errorf("cycle detected: topological sort visited %d of %d nodes", len(order), len(g.Nodes))
	}

	g.Order = order
	return g, nil
}
